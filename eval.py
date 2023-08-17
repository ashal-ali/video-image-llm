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
from evals.templates.imagenet import CLASSNAMES#, TEMPLATES

# pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
from utils.util import state_dict_data_parallel_fix

from imagenetv2_pytorch import ImageNetV2Dataset

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
use_clip = True

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


def zeroshot_classifier(classnames, templates):
    if not use_clip:
        tokenizer = TextProcessor()
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

def get_score(model, loader):
    zeroshot_weights = zeroshot_classifier(CLASSNAMES, TEMPLATES)
    with torch.no_grad():
        top1, top5, top20, n = 0., 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images
            target = target
            
            if USE_CUDA:
                images = images.cuda()
                target = target.cuda()

            # predict
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

    # TODO: Select one of the two models
    model = "ours" # one of "ours", "clip"
    dataset = "imagenet" # one of "imagenet", "ucf101" TODO: imagenetv2

    if model == "ours":
        model = get_model(config)
        use_clip = False
    elif model == "clip":
        model, _ = clip.load("ViT-B/16")
    else:
        raise NotImplementedError

    _, preprocess = clip.load("ViT-B/16")

    # TODO: Select one of the two datasets
    if dataset == "imagenet":
        dataset = ImageNetV2Dataset(transform=preprocess)
    elif dataset == "ucf101":
        dataset = UCF101Dataset(transform=preprocess)
    else:
        raise NotImplementedError
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=8)
    get_score(model, loader)
