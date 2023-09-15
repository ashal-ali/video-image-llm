# Test a given set of captions
# Usage: python test_captions.py <captions_file> 

# Use CLIP to test the caption
import pandas as pd
import sys
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
USE_CUDA = True

if USE_CUDA:
    model = model.cuda()
# Load the captions, read in csv file
df = pd.read_csv(sys.argv[1])

# Drop rows with "IRRELEVANT CAPTION" as the second row
df.drop(df[df['text'] == "IRRELEVANT CAPTION"].index, inplace = True)

# Divide into text and image paths
texts_total = df['text'].tolist()
image_paths = df['img_path'].tolist()

images = []
texts = []
all_correct = 0
batch_size = 128
total_num = len(image_paths)
full_batch_num = 0
full_batch_correct = 0
print(f"Collecting results with batch size {batch_size}")
with torch.no_grad():
    for path, text in tqdm(zip(image_paths, texts_total), total=total_num):
        image = Image.open(path)
        images.append(image)
        texts.append(text)
        if len(images) == batch_size or path == image_paths[-1]:
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            if USE_CUDA:
                inputs = {key: val.cuda() for key, val in inputs.items()}
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
            # Calculate accuracy per batch
            # Get the index of the max log-probability
            pred = probs.argmax(dim=1, keepdim=True)

            if USE_CUDA:
                pred = pred.cpu()

            # Batch is accurate if preds are along the diagonal
            size = pred.size(0)
            correct = pred.eq(torch.arange(size).view(-1, 1))

            if size == batch_size:
                full_batch_num += batch_size
                full_batch_correct += correct.sum().item()

            # Calculate accuracy, convert boolean to int
            correct = correct.sum().item()
            all_correct += correct
            images = []
            texts = []


# Load input
print("Full batch accuracy: {}".format(full_batch_correct / full_batch_num * 100))
print("Total Accuracy: {}".format(all_correct / total_num * 100))

