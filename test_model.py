# Test model to see if it matches the huggingface model
from PIL import Image
import requests
import argparse
from parse_config import ConfigParser
import model.model as module_arch
import torch 

from transformers import CLIPProcessor, CLIPModel

# Get our model's output
def get_our_model(config):
    model = config.initialize('arch', module_arch)
    model.eval()
    return model

# Get Huggingface model output
def get_outputs(our_model):
    with torch.no_grad():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Huggingface model params:", pytorch_total_params)
        #import pdb; pdb.set_trace()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        url = "http://images.cocodataset.org/val2017/000000007386.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

        text_embeds = our_model.compute_text(inputs)
        # Expand image to 5D tensor (b, t, c, h, w)
        video_input = inputs['pixel_values'].unsqueeze(1)
        image_embeds = our_model.compute_video(video_input) 
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        o = our_model.text_model(inputs['input_ids'], inputs['attention_mask'])

        outputs = model(**inputs)
        print("Huggingface text:", outputs['text_embeds'][0][0:10])
        print("Our text:", text_embeds[0][0:10])

        print("Huggingface image:", outputs['image_embeds'][0][0:10])
        print("Our image:", image_embeds[0][0:10])
    opt = torch.optim.Adam(our_model.parameters(), lr=1)
    print("Testing that backward video pass does not change text or image embeddings")
    # b, t, c, h, w
    random_video = torch.randn(1, 4, 3, 224, 224)
    output = our_model.compute_video(random_video)
    toy_loss = torch.sum(output)
    #import pdb; pdb.set_trace()
    opt.zero_grad()
    toy_loss.backward()
    opt.step()

    text_embeds = our_model.compute_text(inputs)
    # Expand image to 5D tensor (b, t, c, h, w)
    video_input = inputs['pixel_values'].unsqueeze(1)
    image_embeds = our_model.compute_video(video_input) 
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    o = our_model.text_model(inputs['input_ids'], inputs['attention_mask'])
    print("Huggingface text:", outputs['text_embeds'][0][0:10])
    print("Our text:", text_embeds[0][0:10])

    print("Huggingface image:", outputs['image_embeds'][0][0:10])
    print("Our image:", image_embeds[0][0:10])





    return outputs

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
    model = get_our_model(config)
    #model = None
    outputs = get_outputs(model)
