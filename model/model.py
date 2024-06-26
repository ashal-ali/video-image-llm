import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, CLIPTokenizer, CLIPProcessor

from base import BaseModel
from model.video_transformer import SpaceTimeTransformer
from model.clip_layers import CLIP_INIT_LAYERS
from utils.util import state_dict_data_parallel_fix


class FrozenInTime(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")
        if "clip" in text_params['model']:
            # TODO: Add other clip text encoders
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.text_model = clip_model.text_model
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            #clip_config = CLIPTextConfig.from_pretrained("openai/clip-vit-base-patch16")
            #self.text_model = CLIPTextModel(clip_config).text_model
        else:    
            self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()
        text_frozen = text_params['text_frozen']
        if text_frozen:
            for p in self.text_model.parameters():
                p.requires_grad = False

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            vit_init = video_params.get('vit_init', 'imagenet-21k')
            freeze_first_frame = video_params.get('freeze_first_frame', False)
            vit_frozen = video_params.get('vit_frozen', False)
            patch_drop_rate = video_params.get('patch_drop_rate', 0.0)
            if arch_config == 'base_patch16_224':
                vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                model = SpaceTimeTransformer(
                            num_frames=num_frames,
                            time_init=time_init,
                            attention_style=attention_style,
                            patch_drop_rate=patch_drop_rate,
                            freeze_first_frame=freeze_first_frame,
                        )
            elif arch_config == 'base_patch16_clip_224':
                vit_model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=pretrained)
                model = SpaceTimeTransformer(
                            num_frames=num_frames,
                            time_init=time_init,
                            attention_style=attention_style,
                            patch_drop_rate=patch_drop_rate,
                            freeze_first_frame=freeze_first_frame,
                            clip=True
                        )
            else:
                raise NotImplementedError 
            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                #import pdb; pdb.set_trace()
                vit_checkpoint = vit_model.state_dict()
                ckpt_vals = model.load_state_dict(vit_checkpoint, strict=False)
                if "clip" in arch_config:
                    # Manually load CLIP weights with different names
                    nn.init.zeros_(model.patch_embed.proj.bias) # TODO: Change bias to be False and add flag during init
                    for block in model.blocks:
                        nn.init.ones_(block.norm3.weight)
                        nn.init.zeros_(block.norm3.bias)
                    # model.patch_embed.proj.weight = vit_checkpoint['patch_embed.proj.weight']
            if vit_frozen:
                model.pos_embed.requires_grad = False
                model.cls_token.requires_grad = False
            if vit_frozen and "clip" in arch_config:
                for name, layer in model.named_children():
                    if name == "blocks":
                        for block in layer:
                            for b_name, b_layer in block.named_children():
                                if b_name in CLIP_INIT_LAYERS:
                                    for p in b_layer.parameters():
                                        p.requires_grad = False
                    elif name in CLIP_INIT_LAYERS:
                        print(f"Freezing {name}")
                        for p in layer.parameters():
                            p.requires_grad = False
                    else:
                        print(f"Skipping {name} for freezing")
                    
            #import pdb; pdb.set_trace()
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding  
        if "clip" in text_params['model'] and "clip" in arch_config:
            txt_proj = clip_model.text_projection
            #vid_proj = nn.Identity()
            vid_proj = clip_model.visual_projection
            if vit_frozen:
                for p in vid_proj.parameters():
                    p.requires_grad = False
            if text_frozen:
                for p in txt_proj.parameters():
                    p.requires_grad = False

            # vid_proj set to identity       
        elif projection == 'minimal':
            txt_ftr_dim = self.text_model.config.hidden_size
            txt_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(txt_ftr_dim, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        elif projection == '':
            print("Using identity projection")
            #txt_proj = nn.Identity()
            #if "clip" in text_params['model']:
            #txt_ftr_dim = 512
            #print("OUTPUT change for text:", txt_ftr_dim, projection_dim)
            #print("OUTPUT of video model:", ftr_dim, projection_dim)
            #txt_proj = nn.Identity() 
            #nn.Sequential(nn.ReLU(),
            #                nn.Linear(txt_ftr_dim, ftr_dim),
            #                )
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)

    def set_device(self, device):
        self.device = device

    def forward(self, data, return_embeds=True):
        text_data = data['text']
        video_data = data['video']

        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data):
        if "clip" in self.text_params['model']:
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    # Function wrapper compatibility with OpenAI's CLIP
    def encode_text(self, text_data): 
        return self.compute_text(text_data)
    
    # Function wrapper compatibility with OpenAI's CLIP
    def encode_image(self, image_data): 
        video_data = image_data.unsqueeze(1)
        return self.compute_video(video_data)

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict


def sim_matrix(a, b, temperature=torch.Tensor(1), eps=1e-8):
    """
    added eps for numerical stability
    """
    scale = torch.exp(temperature)
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = scale * torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def compute_similarity(a, b, a_mask=None, b_mask=None, style='single', eps=1e-8, return_raw=False, temp=0.5):
    if style == 'single':
        sim = sim_matrix(a, b, eps=eps)
        return sim, sim.t()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    pass
