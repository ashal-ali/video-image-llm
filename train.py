import argparse
import collections
import os

import transformers
from transformers import CLIPTokenizer
from sacred import Experiment

import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualizer as module_vis
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import replace_nested_dict_item
from neptune.integrations.sacred import NeptuneObserver
import neptune
import wandb
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

ex = Experiment('train')

def get_dist_args():
    envvars = [
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "NODE_COUNT",
        "HOSTNAME",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_SOCKET_IFNAME",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "AZ_BATCHAI_MPI_MASTER_NODE",
    ]
    args = dict(gpus_per_node=torch.cuda.device_count())
    missing = []
    for var in envvars:
        if var in os.environ:
            args[var] = os.environ.get(var)
            try:
                args[var] = int(args[var])
            except ValueError:
                pass
        else:
            missing.append(var)
    print(f"II Args: {args}")
    if missing:
        print(f"II Environment variables not set: {', '.join(missing)}.")
    return args

# Disable Cuda for debugging
DEBUG = False # Run all processes on single gpu for debugging
if DEBUG:
    print("====!!  DEBUGGING WITH SINGLE GPU  !!===")
import pdb

def ddp_setup(rank, world_size):
    print(f"SETUP FOR RANK {rank}")
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12323'
    if DEBUG:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        #torch.cuda.set_device('cpu')
        #print("Initializing process group with gloo backend")
        #init_process_group(backend='nccl', rank=rank, world_size=world_size)
        #print("init done!")
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(rank)
    #init_process_group(backend='nccl', rank=rank, world_size=world_size)


#def ddp_run(rank, world_size, config):
def ddp_run(global_rank, local_rank, world_size, config):
    #dist.init_process_group(backend='nccl')
    #rank = dist.get_rank()
    #torch.cuda.set_device(rank)
    # TODO: init process with random seed?
    #ddp_setup(rank, world_size)
    if global_rank == 0:
        wandb.login(key="03ba3395fe1ce3201a18226899de14ea958c25cc", force=True)
        wandb.init(project="video-image-llm", config=config._config, dir="/tmp/wandb_new/")

    logger = config.get_logger('train')
    print(f"=== DDP RUN on rank: {global_rank}===")

    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    # build tokenizer
    if "clip" in config['arch']['args']['text_params']['model']:
        tokenizer = CLIPTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                           TOKENIZERS_PARALLELISM=False)
    
    # setup data_loader instances
    # TODO: Change to DistributedSampler
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
    print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')
    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', transformers, trainable_params)
    lr_scheduler = None
    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    if config['trainer']['neptune']:
        writer = ex
    elif config['trainer']['wandb']:
        writer = wandb.log
    else:
        writer = None
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      visualizer=visualizer,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=config['trainer']['max_samples_per_epoch'],
                      debug=DEBUG)
    trainer.train()
    destroy_process_group()
    return


@ex.main
def run():
    logger = config.get_logger('train')
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    # TODO: improve Create identity (do nothing) visualiser?
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None

    # build tokenizer
    if "clip" in config['arch']['args']['text_params']['model']:
        tokenizer = CLIPTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                           TOKENIZERS_PARALLELISM=False)

    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
    print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')
    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', transformers, trainable_params)
    lr_scheduler = None
    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    if config['trainer']['neptune']:
        writer = ex
    elif config['trainer']['wandb']:
        writer = wandb.log
    else:
        writer = None
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      visualizer=visualizer,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=config['trainer']['max_samples_per_epoch'])
    trainer.train()


def init_dataloaders(config, module_data):
    """
    We need a way to change split from 'train' to 'val'.
    """

    # Use CLIP pixel normalization if using a CLIP model
    tsfm_params = {"use_clip_norm": False}
    if "clip" in config["arch"]["args"]["video_params"]["arch_config"]:
        tsfm_params["use_clip_norm"] = True
    
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # then its a single dataloader
        config['data_loader']['args']['tsfm_params'] = tsfm_params
        data_loader = [config.initialize("data_loader", module_data)]
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'split', 'val')
        valid_data_loader = [config.initialize("data_loader", module_data)]
    elif isinstance(config["data_loader"], list):
        # then its a list of dataloaders
        for dl_cfg in config['data_loader']:
            dl_cfg['args']['tsfm_params'] = tsfm_params
        data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                       range(len(config['data_loader']))]
        new_cfg_li = []
        for dl_cfg in config['data_loader']:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', 'val')
            new_cfg_li.append(dl_cfg)
        config._config['data_loader'] = new_cfg_li
        valid_data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                             range(len(config['data_loader']))]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    args.add_argument('-w', '--wandbapi', default=None, type=str,
                        help='wandb api key (default: None)') 
    args.add_argument('-l', '--local_rank', default=0, type=int,
                        help='local rank for distributed training (default: 0)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]    

    args_raw = args.parse_args()
    config = ConfigParser(args, options)
    ex.add_config(config._config)



    if config['trainer']['neptune']:
        # delete this error if you have added your own neptune credentials neptune.ai
        #raise ValueError('Neptune credentials not set up yet.')
        run = neptune.init_run(
            project="zanedurante/video-image-llm",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMDEzZWM5ZS03NzhjLTQ1NmQtOWM5Mi1hZjRjYjBiZjg5ZTcifQ==",
        )
        ex.observers.append(NeptuneObserver(run=run))
        ex.run()
    try:
        strat = config['strat']
    except:
        print("Strategy not specified, defaulting to DataParallel")
        strat = 'dp'
    if strat == 'ddp':
        gpus_per_node = torch.cuda.device_count()
        dist_args = get_dist_args()
        world_size = dist_args.get("WORLD_SIZE")
        gpu_rank = args_raw.local_rank
        node_rank = dist_args.get("NODE_RANK")
        if node_rank is None:
            node_rank = 0 # Add support for single gpu dist training with same launcher
        global_rank = node_rank * gpus_per_node + gpu_rank
        dist.init_process_group(
            backend='nccl', rank=global_rank, world_size=world_size
        )
        ddp_run(global_rank, gpu_rank, world_size, config)
        #mp.spawn(ddp_run, args=(world_size, config), nprocs=world_size) # Use DistributedDataParallel
    else:
        wandb.login(key="03ba3395fe1ce3201a18226899de14ea958c25cc", force=True)
        wandb.init(project="video-image-llm", config=config._config)
        run() # Use DataParallel
