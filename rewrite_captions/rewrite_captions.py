import requests

PROMPT = """Rewrite the following alt text captions into complete sentences.  Be sure that the resulting sentences sound natural and remove information irrelevant to how the image looks visually.  Flag an alt text caption with IRRELEVANT CAPTION seems to not be relevant to what is visually present in the image.

For example:
INPUT: "studio shot of a black smartphone , with mp , f"
OUTPUT:  An image of a black smartphone

INPUT: person says the next race should be better after best - ever finish
OUTPUT: IRRELEVANT CAPTION

INPUT:  "get ready , because you 're about to learn the secrets of truly living well ."
OUTPUT: IRRELEVANT CAPTION

INPUT: person took this photo of the full moon peeking through the trees .
OUTPUT: A photo of the full moon peeking through the trees.

INPUT: fresh water pouring in a glass on white background in sequence -- stock photo #
OUTPUT: Fresh water pouring in a glass on a white background.

INPUT: i love this so much !
OUTPUT: IRRELEVANT CAPTION

INPUT: INSERT_CAPTION_HERE 
OUTPUT: 
"""

urls = [
    'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2023-03-15-preview',
    'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview',
#    'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2023-05-15',
#    'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-05-15',
#    'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2023-06-01-preview',
#    'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-06-01-preview',
]

url = 'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2023-05-15'

api_keys = ['api-key'] # not begin with sk-

global api_key_idx
global url_idx

def next_api_key():
    global api_key_idx
    api_key_idx = (api_key_idx + 1) % len(api_keys)
    return api_keys[api_key_idx]

def next_url():
    global url_idx
    url_idx = (url_idx + 1) % len(urls)
    return urls[url_idx]

api_key = 'api-key' # not begin with sk-  
headers = {'Content-Type': 'application/json', 'api-key': api_key}  

def get_prompt(caption):
    return PROMPT.replace("INSERT_CAPTION_HERE", caption)

def get_request(caption):
    prompt = get_prompt(caption)
    return {
        "messages": [  
            {"role": "system", "content": "You are a helpful assistant."},  
            {"role": "user", "content": prompt},
        ],  
        "max_tokens": 500,
        "temperature": 0.0,
        "n":1
    }

def prompt_gpt4(new_caption):
    data = get_request(new_caption)
    return requests.post(url, json=data, headers=headers).json()['choices'][0]['message']['content']

api_key_idx = 0
url_idx = 0
# url = 'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-06-01-preview' 
# urls[url_idx]#
# test = prompt_gpt4("test")
# print(test)

# Load captions from 1k_orig.csv
import pandas as pd
from tqdm import tqdm
import time
df = pd.read_csv("train.csv")
texts = df['text'].tolist()
start_idx = 15731 + 21268 + 493 + 204 # Add previous indices + end line from most recent file

from glob import glob
files = glob("rewrites/new_train_captions_*.txt")
start_idx = 0
for file in files: # Calculate start_idx instead
    with open(file, "r") as f:
        new_texts = f.readlines()
        start_idx += len(new_texts)

end_idx = 0
files = glob("rewrites/reversed_new_train_captions_*.txt")
for file in files: # Calculate end_idx instead
    with open(file, "r") as f:
        new_texts = f.readlines()
        end_idx += len(new_texts)

print(texts[start_idx-5:start_idx+5])
print(f"Previous caption: {texts[start_idx-1]}")
print(f"Starting with caption: {texts[start_idx]}")
if end_idx > 0:
    texts = texts[start_idx:-end_idx]
else:
    texts = texts[start_idx:]
new_captions = []
number_of_new_captions = 0
for caption in tqdm(texts):
    num_consec_fails = 0
    success = False
    while not success:
        url = next_url()
        api_key = next_api_key()
        try:
            new_caption = prompt_gpt4(caption)
            success = True
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            num_consec_fails += 1
            print(e)
            time.sleep(0.2)
            if num_consec_fails > 6: # Failing on each model
                time.sleep(1)
            if num_consec_fails > 100:
                exit()
    new_captions.append(new_caption)
    number_of_new_captions += 1
    with open(f"rewrites/new_train_captions_{start_idx}.txt", "a") as f:
        f.write(new_caption + "\n")

# Save new captions to 1k_new.csv
df['text'] = new_captions
df.to_csv("1k_new.csv", index=False)