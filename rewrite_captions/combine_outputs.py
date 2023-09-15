from glob import glob

files = glob('rewrites_chatgpt/*.txt') # Rewrites for webvid10m

all_lines = []
for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
        print("File: ", file, " has ", len(lines), " lines")
        all_lines.extend(lines)
    
with open('rewrites_chatgpt/all_rewrites.txt', 'w') as f:
    for line in all_lines:
        f.write(line)
print(f"WebVid10M has {len(all_lines)} rewrites")

files = glob("rewrites/new_train_captions_*.txt") # Rewrites for cc3m

all_lines = []
for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
        all_lines.extend(lines)

with open('rewrites/all_rewrites.txt', 'w') as f:
    for line in all_lines:
        f.write(line)
print(f"CC3M has {len(all_lines)} rewrites")


# Create new csv file for webvid and cc3m
import pandas as pd

# Add rewrites to cc3m
with open('rewrites/all_rewrites.txt', 'r') as f:
    rewrites = f.readlines()

cc3m_df = pd.read_csv('train.csv')[:len(rewrites)]
cc3m_df_copy = cc3m_df.copy()

cc3m_df['text'] = rewrites
irr_caption_idxs = [idx for idx,s in enumerate(rewrites) if "IRRELEVANT CAPTION" in s]
total_captions_idxs = list(range(len(rewrites)))
good_caption_idxs = list(set(total_captions_idxs) - set(irr_caption_idxs))
print(max(irr_caption_idxs))
print(min(irr_caption_idxs))
cc3m_df = cc3m_df.loc[good_caption_idxs]
cc3m_df_copy = cc3m_df.loc[good_caption_idxs]
cc3m_df.to_csv('/mnt/datasets_mnt/cc3m/final_split/train_rewrites.csv', index=False)
cc3m_df_copy.to_csv('/mnt/datasets_mnt/cc3m/final_split/train_rewrites_equiv.csv', index=False)

print(f"CC3M has {len(cc3m_df)} entries now")
print(f"CC3M equiv has {len(cc3m_df_copy)} entries now")

# Add rewrites to webvid
with open('rewrites_chatgpt/all_rewrites.txt', 'r') as f:
    rewrites = f.readlines()

webvid_df = pd.read_csv('results_10M_train.csv')[:len(rewrites)]
webvid_df_copy = webvid_df.copy()

webvid_df['text'] = rewrites
webvid_df.to_csv('/mnt/datasets_mnt/webvid10m/metadata/results_rewrites_train.csv', index=False)
webvid_df_copy.to_csv('/mnt/datasets_mnt/webvid10m/metadata/results_rewrites_equiv_train.csv', index=False)

print(f"WebVid10M has {len(webvid_df)} entries now")