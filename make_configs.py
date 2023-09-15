import sys
import os
import json

# Update the following function to add the grid search parameters 

PREEMPTIBLE = True

def get_params():
    return {
#        "arch.args.video_params.patch_drop_rate": [0.0, 0.25, 0.5, 0.75, 0.9],
#        "<sync>data_loader.0.args.batch_size": [64, 128, 256],
#        "optimizer.args.lr": [1e-4, 1e-5],
        "freeze_first_frame": [False, True],
        "temperature": [True],
        "n_node": [4],
    }

def num_param_combos(params):
    num = 1
    for k, v in params.items():
        num *= len(v)
    return num

def get_param_combo(params, idx):
    combo = {}
    for k, v in params.items():
        combo[k] = v[idx % len(v)]
        idx //= len(v)
    return combo

def get_run_name(combo, exp_name):
    name = exp_name
    for k, v in combo.items():
        name += f"_{k}-{v}"
    return name

def set_val(config, key, val):
    if key == "patch_drop_rate":
        config["arch"]["args"]["video_params"]["patch_drop_rate"] = val
    elif key == "batch_size":
        for i in range(len(config["data_loader"])):
            config["data_loader"][i]["args"]["batch_size"] = val
    elif key == "lr":
        config["optimizer"]["args"]["lr"] = val
    elif key == "text_frozen":
        config["arch"]["args"]["text_params"]["text_frozen"] = val
    elif key == "n_node":
        config["n_node"] = val
    elif key == "n_gpu":
        config["n_gpu"] = val
    elif key == "freeze_first_frame":
        config["arch"]["args"]["video_params"]["freeze_first_frame"] = val
    elif key == "temperature":
        config["loss"]["args"]["temperature"] = val
    else:
        raise ValueError(f"Unknown key {key}, need to implement mapping in set_val")

if len(sys.argv) < 3:
    print("Usage: python make_configs.py <orig_config_filename> <exp_name> <optional: run_location (e.g. local, hai5, etc.)>")
    sys.exit(1)

orig_config_filename = sys.argv[1]
exp_name = sys.argv[2]
run_local = False
deploy_loc = None
if len(sys.argv) > 3:
    if sys.argv[3] == "local":
        run_local = True
    else:
        deploy_loc = sys.argv[3]


with open(orig_config_filename, "r") as f:
    orig_config = json.load(f)
#orig_config = read_json(orig_config_filename)

CONFIG_DIR = f"configs/{exp_name}"
os.makedirs(CONFIG_DIR, exist_ok=True)

params = get_params()
num_combos = num_param_combos(params)

for i in range(num_combos):
    combo = get_param_combo(params, i)
    run_name = get_run_name(combo, exp_name)
    config = orig_config.copy()
    config["name"] = run_name
    for k, v in combo.items():
        set_val(config, k, v)
    config["trainer"]["save_dir"] = f"/mnt/datasets_mnt/output/exps/{exp_name}/{run_name}" 
    config_filename = f"{CONFIG_DIR}/{run_name}.json"

    with open(config_filename, "w") as f:
        json.dump(config, f, indent=4)

    #write_json(config, config_filename)

if run_local:
    for i in range(num_combos):
        combo = get_param_combo(params, i)
        run_name = get_run_name(combo, exp_name)
        config_filename = f"{CONFIG_DIR}/{run_name}.json"
        print(f"Running {config_filename}")
        os.environ["WANDB_NAME"] = run_name
        os.system(f"python train.py --config {config_filename}")

def get_amlt_job(combo, exp_name):
    name = get_run_name(combo, exp_name)
    # Defaults
    n_gpu = 16
    n_node = 1
    if "n_gpu" in combo:
        n_gpu = combo["n_gpu"]
    if "n_node" in combo:
        n_node = combo["n_node"]

    job = f"""
- name: {name}
  sku: {n_node}xG{n_gpu}
  preemptible: false
  process_count_per_node: 1
  command:
  - python -m torch.distributed.launch --nnodes={n_node} --nproc_per_node={n_gpu} train.py --config configs/{exp_name}/{run_name}.json

    """
    if PREEMPTIBLE:
        job.replace('false', 'true')
    return job


if deploy_loc is not None:
    # Write amlt script
    base_script = f"""
description: {exp_name}

target:
  service: amlk8s
  name: itphyperdgx2cl1 
  vc: {deploy_loc}

environment:
  image: images/video-image-llm:v7 
  username: zaneml
  registry: zaneml.azurecr.io
  setup:
    - pip install wandb
    - export GIT_PYTHON_REFRESH=quiet
    - mkdir /mnt/datasets_mnt/

code:
  local_dir: $CONFIG_DIR/video-image-llm
storage:
  datasets: 
    storage_account_name: zane
    container_name: data
    mount_dir: /mnt/datasets_mnt
    mount_options: ["-o", "attr_timeout=240", "-o", "entry_timeout=240", "-o", "negative_timeout=120", "-o", "allow_other"]
    local_dir: /home/t-zadurante/mnt/datasets_mnt

jobs:
    """
    for i in range(num_combos):
        combo = get_param_combo(params, i)
        base_script += get_amlt_job(combo, exp_name)

    print(base_script)
    base_script.replace("/video-image-llm", "$CONFIG_DIR/video-image-llm")
    os.system(f'cd ../; echo "{base_script}" > {exp_name}.yaml')
    print(f'Run: cd ../; amlt run {exp_name}.yaml')
