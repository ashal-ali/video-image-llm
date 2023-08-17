import numpy as np
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging

from templates.imagenet import TEMPLATES, CLASSNAMES

# pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
# TODO: Add implementation for my models

from imagenetv2_pytorch import ImageNetV2Dataset

# Eval code largely taken from: https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/notebooks/Prompt_Engineering_for_ImageNet.ipynb

model, preprocess = clip.load("ViT-L/14")

print("Loaded model!")

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


zeroshot_weights = zeroshot_classifier(CLASSNAMES, TEMPLATES)


images = ImageNetV2Dataset(transform=preprocess)
loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

with torch.no_grad():
    top1, top5, top20, n = 0., 0., 0., 0.
    for i, (images, target) in enumerate(tqdm(loader)):
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