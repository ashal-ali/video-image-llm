import json

# Specify the path to the JSON file
file_path = 'i21k_vit-frozen-distilbert-frozen.json'

# Read the JSON data from the file
with open(file_path, 'r') as file:
    data = json.load(file)

# create key/val for each of the options
vit_arch = {"i21k_vit":"base_patch16_224", "clip_vit":"base_patch16_clip_224"}
vit_frozen = {"frozen":True, "init":False}
text_arch = {"distilbert":"distilbert-base-uncased", "clip_text":"openai/clip-vit-base-patch16"}
text_frozen = {"frozen":True, "init":False}

# create a list of all the options
all_names = []
for vit_name, vit_val in vit_arch.items():
    for vit_frozen_str, vit_frozen_val in vit_frozen.items():
        for text_arch_str, text_arch_val in text_arch.items():
            for text_frozen_str, text_frozen_val in text_frozen.items():
                # Create - separated name
                name = f"{vit_name}-{vit_frozen_str}-{text_arch_str}-{text_frozen_str}.json"
                all_names.append(name)
                # Create copy of json file
                new_data = data.copy()
                
                # Set the new values
                new_data['arch']['args']['video_params']['arch_config'] = vit_val
                new_data['arch']['args']['video_params']['vit_frozen'] = vit_frozen_val
                new_data['arch']['args']['text_params']['model'] = text_arch_val
                new_data['arch']['args']['text_params']['text_frozen'] = text_frozen_val

                # Save the new json file
                with open(name, 'w') as outfile:
                    json.dump(new_data, outfile, indent=4)

# Create amulet script from template
new_vals = """- name: TEMPLATE-NO-JSON
  sku: G16
  command:
  - python train.py --config configs/model_search/TEMPLATE
"""

# Read amulet script template
with open('amlt_template.yaml', 'r') as file:
    lines = file.readlines()

# Create amulet script with every option
for name in all_names:
    # Add new values to template
    to_append = new_vals.replace("TEMPLATE", name).replace(".json-NO-JSON", "")

    # Append values to lines
    lines.append(to_append)

with open('new_run.yaml', 'w') as file:
    file.writelines(lines)
    



#print(data['arch']['args']['video_params']['arch_config']) # base_patch16_224 or base_patch16_clip_224
#print(data['arch']['args']['video_params']['vit_frozen']) # True or False

#print(data['arch']['args']['text_params']['model']) # distilbert-base-uncased or openai/clip-vit-base-patch16
#print(data['arch']['args']['text_params']['text_frozen']) # True or False