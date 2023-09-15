import numpy as np
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging
import argparse
import model.model as module_arch
from parse_config import ConfigParser
from transformers import CLIPProcessor
from transformers.tokenization_utils_base import BatchEncoding
 #, TEMPLATES

# pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
from utils.util import state_dict_data_parallel_fix

from imagenetv2_pytorch import ImageNetV2Dataset
from data_loader.UCF101 import UCF101Dataset

from viclip.viclip import ViCLIP

USE_CUDA = True
USE_ENSEMBLING = False

if USE_ENSEMBLING:
    from evals.templates.imagenet import TEMPLATES
else:
    from evals.templates.base import IMAGE_TEMPLATES as TEMPLATES

# Eval code largely taken from: https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/notebooks/Prompt_Engineering_for_ImageNet.ipynb

# model, preprocess = clip.load("ViT-L/14")

# Take text as input like Openai's CLIP
# Process data into form expected by Huggingface's CLIP
global use_clip
use_clip = "True"

global num_frames
num_frames = 1

class TextProcessor:
    def __init__(self):
        self.process = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    def __call__(self, text):
        return self.process(text=text, return_tensors="pt", padding=True)

def get_model(config):
    model = config.initialize('arch', module_arch)
    if config.resume:
        print(f"Loading model from checkpoint: {config.resume}")
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    if USE_CUDA:
        model.cuda()
    print("Loaded model!")
    return model

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def zeroshot_classifier(classnames, templates, model):
    if use_clip == "False":
        tokenizer = TextProcessor()
    elif use_clip == "ViCLIP":
        tokenizer = model.tokenizer
    else:
        tokenizer = clip.tokenize
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts)# .cuda() #tokenize
            if USE_CUDA:
                if isinstance(texts, BatchEncoding):
                    for key in texts.keys():
                        texts[key] = texts[key].cuda()
                else:
                    texts = texts.cuda()
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        if USE_CUDA:
            zeroshot_weights = zeroshot_weights.cuda()
    return zeroshot_weights

def get_score(model, loader, cls_names=None):
    global num_frames
    if cls_names is None:
        zeroshot_weights = zeroshot_classifier(CLASSNAMES, TEMPLATES, model)
    else:
        zeroshot_weights = zeroshot_classifier(cls_names, TEMPLATES, model)
    with torch.no_grad():
        top1, top5, top20, n = 0., 0., 0., 0.
        for i, vals in enumerate(tqdm(loader)):
            images, target = vals
            images = images
            target = target
            
            if USE_CUDA:
                images = images.cuda()
                target = target.cuda()

            # predict
            if "frozen" in str(type(model)).lower() and num_frames > 1: # ours
                image_features = model.compute_video(images)
            else:  # clip or single frame ours
                image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5, acc20 = accuracy(logits, target, topk=(1, 5, 20))
            top1 += acc1
            top5 += acc5
            top20 += acc20
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 
    top20 = (top20 / n) * 100 

    print(f"Top-1 accuracy: {top1:.3f}")
    print(f"Top-5 accuracy: {top5:.3f}")
    print(f"Top-20 accuracy: {top20:.3f}")
    return top1, top5, top20

# Assumes model is already loaded and is ours
def run_ucf101_eval(model):
    global use_clip, num_frames
    use_clip = "False"
    num_frames = 4 # TODO: Set to num frames from model params
    from evals.templates.ucf101 import CLASSNAMES
    from data_loader.UCF101 import UCF101Dataset
    ds = UCF101Dataset(num_frames=num_frames)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=8)
    if "DistributedDataParallel" in str(type(model)): # DistributedDataParallel fix (use model.module)
        results = get_score(model.module, loader, cls_names=CLASSNAMES)
    else:
        print(f"Not DDP model, using model of type {str(type(model))} directly")
        results = get_score(model, loader, cls_names=CLASSNAMES)
    return results

def run_imagenetv2_eval(model):
    global use_clip, num_frames
    use_clip = "False"
    num_frames = 1 # image dataset
    from evals.templates.imagenet import CLASSNAMES
    from imagenetv2_pytorch import ImageNetV2Dataset
    _, preprocess = clip.load("ViT-B/16")
    ds = ImageNetV2Dataset(transform=preprocess)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=8)
    if "DistributedDataParallel" in str(type(model)): # DistributedDataParallel fix (use model.module)
        results = get_score(model.module, loader, cls_names=CLASSNAMES)
    else:
        print(f"Not DDP model, using model of type {str(type(model))} directly")
        results = get_score(model, loader, cls_names=CLASSNAMES)
    return results

if __name__ == '__main__':
    #url = "http://images.cocodataset.org/val2017/000000007386.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
    config = ConfigParser(args)
    n_frames = config["arch"]["args"]["video_params"]["num_frames"]

    # TODO: Select one of the two models
    model_name = "ours" # one of "ours", "clip", "viclip"
    dataset_name = "ucf101" # one of "imagenet", "ucf101", TODO: imagenetv2 and kinetics400

    if model_name == "ours":
        model = get_model(config)
        use_clip = "False"
    elif model_name == "clip":
        model, _ = clip.load("ViT-B/16")
    elif model_name == "viclip":
        model = ViCLIP(pretrain="viclip/ViClip-InternVid-10M-FLT.pth")
        use_clip = "ViCLIP"

    else:
        raise NotImplementedError

    _, preprocess = clip.load("ViT-B/16")

    # TODO: Select one of the two datasets
    if dataset_name == "imagenet":
        from evals.templates.imagenet import CLASSNAMES
        dataset = ImageNetV2Dataset(transform=preprocess)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=8)
        get_score(model, loader)
    elif dataset_name == "ucf101":
        from evals.templates.ucf101 import CLASSNAMES
        if model_name == "ours" or model_name == "viclip":
            preprocess=None
            kind="video"
        elif model_name == "clip":
            preprocess=preprocess
            kind="image"
        dataset = UCF101Dataset(tfms=preprocess, kind=kind, n_frames=8)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=8)
        get_score(model, loader)
    elif dataset_name == "kinetics":
        from evals.templates.kinetics400 import CLASSNAMES
        if model_name == "ours" or model_name == "viclip":
            preprocess=None
            kind="video"
        elif model_name == "clip":
            print("CLIP for kinetics has not been implemented yet.")
            raise NotImplementedError
        dataset = Kinetics(tfms=preprocess, split="val")
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=8)
        get_score(model, loader)
    else:
        raise NotImplementedError
    
    
