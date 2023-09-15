import requests
import time
import pandas as pd
from glob import glob
from tqdm import tqdm

PROMPT = """Rewrite the following stock footage descriptors into complete sentences.  Be sure that the resulting sentences sound natural and remove specifics about cameras.

For example:
INPUT: 3d render of inky injections into water with luma matte. blue ink on white background 5
OUTPUT: Blue ink injections onto a white background.

INPUT: "Swimming in the pool ,slow motion 120 fps,handheld camera balanced steady shot " 
OUTPUT: A person swimming in the pool.

INPUT: Aerial drone isle of wight needles england travel sunrise
OUTPUT: The sun rises over the Isle of Wight Needles in England.

INPUT: CAPTION
OUTPUT: """

urls = [
    'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2023-03-15-preview',
    'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview',
]

url = 'https://gcrgpt4aoai4c.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2023-05-15'

api_keys = ['api-key'] # no sk-

global api_key_idx
global url_idx

# logging info to choose optimal endpoint
end2stats = {key: {"total_time": 0, "num_succ_calls": 0, "num_fails": 0} for key in urls}

global endpoint_index
endpoint_index = 0

def get_next_endpoint():
    global endpoint_index
    endpoint_index += 1
    endpoint_index = endpoint_index % len(urls)
    return urls[endpoint_index]

global api_key_index
api_key_index = 0

def get_next_api_key():
    global api_key_index
    api_key_index += 1
    api_key_index = api_key_index % len(api_keys)
    return api_keys[api_key_index]


def get_prompt(caption):
    return PROMPT.replace("CAPTION", caption)

def prompt_chatgpt(caption, endpoint="gpt-3.5-turbo"):
    prompt = get_prompt(caption)
    return requests.post(url, json=data, headers=headers).json()['choices'][0]['message']['content']

#print(prompt_chatgpt("Usd cash macro view. green 100 dollar cash."))
