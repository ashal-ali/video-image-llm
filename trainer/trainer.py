import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from base import BaseTrainer
from model.model import sim_matrix
from utils import inf_loop
import random

import torch
from torch import Tensor
torch.autograd.set_detect_anomaly(True) # TODO: Remove for optimal performance
import os

from eval import run_imagenetv2_eval, run_ucf101_eval

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000, debug=False, global_rank=0):
        super().__init__(model, loss, metrics, optimizer, config, writer, debug=debug)
        self.config = config
        self.wandb = config['trainer']['wandb']
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min(len(x) for x in data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.global_rank = global_rank
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.total_batch_sum = sum(x.batch_size for x in self.data_loader)
        self.tokenizer = tokenizer
        if 'global' in self.loss.__class__.__name__.lower():
            self.global_loss = True
        else:
            self.global_loss = False
        self.max_samples_per_epoch = max_samples_per_epoch
        self.temperature = torch.tensor(1.0)
        if self.config["loss"]["args"]["temperature"]:
            self.temperature = nn.Parameter(torch.tensor([np.log(1/0.07)]))
        self.temperature = self.temperature.to(self.device) 
        nn.Parameter(torch.tensor([np.log(1/0.07)]))
        if self.ddp:
            self.accumulation_steps = 1 # TODO: Set in config

    def _eval_metrics(self, output):
        log_dict = {}
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                if self.wandb:
                    if self.global_rank == 0:
                        log_dict['{}'.format(metric.__name__)] = acc_metrics[i]
                else:
                    self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        if self.wandb and self.global_rank == 0:
            self.writer(log_dict)
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        # Note: Due to the large dataset size, the code has been modified as follows:
        # max_samples_per_epoch: maximum number of samples before validation
        # Here, epoch is rather used as a number of iterations rather than viewing each of the data once
        
        # Shuffle order of data at the beginning of the epoch
        # Thus, when we train for each epoch new data is seen every time
        for dl in self.data_loader:
            dl.dataset.shuffle_order()
            if self.ddp:
                dl.sampler.set_epoch(epoch)
        #import pdb; pdb.set_trace()
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_iterations = self.max_samples_per_epoch // self.total_batch_sum + 1
        with tqdm(zip(*self.data_loader), desc=f"Training epoch {epoch}", total=total_iterations) as progress:
            for batch_idx, data_li in enumerate(progress):
                if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                    break
                for dl_idx, data in enumerate(data_li):
                    # then assume we must tokenize the input, e.g. its a string
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                      truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)
                    count_dict = {}
                    #for name, param in self.model.named_parameters():
                    #    if 'weight' in name:
                    #        if param.grad is not None:
                    #            count_dict[name] = torch.zeros(param.grad.shape)
                    self.optimizer.zero_grad()
                    # for debugging
                    #import pdb; pdb.set_trace()
                    #random_int = random.randint(0, 10000)
                    #rank = torch.distributed.get_rank()
                    #torch.save(rank, f"/mnt/datasets_mnt/debug/fail_rank_{random_int}.pt")
                    #torch.save(data['text'], f"/mnt/datasets_mnt/debug/fail_text_{random_int}.pt")
                    
                    # For debugging send to cpu
                    #data['text'] = {key: val.cpu() for key, val in data['text'].items()}
                    #data['video'] = data['video'].cpu()
                    #self.model.cpu()(data)
                    #s = str(data['text'])
                    #with open("/mnt/datasets_mnt/debug/0fail_text_all.txt", "a") as f:
                    #    f.write(s + "\n")
                    text_embeds, video_embeds = self.model(data)
                    #if self.ddp and loss_type == 'global':
                    #if False:
                        # Synchronize outputs
                        # import pdb; pdb.set_trace()
                        #all_text_embeds = [torch.zeros_like(text_embeds) for _ in range(torch.distributed.get_world_size())]
                        #all_video_embeds = [torch.zeros_like(video_embeds) for _ in range(torch.distributed.get_world_size())]
                        
                        #torch.distributed.all_gather(all_text_embeds, text_embeds)
                        #torch.distributed.all_gather(all_video_embeds, video_embeds)
                    #    combined_text, combined_videos = AllGatherGrad.apply(text_embeds), AllGatherGrad.apply(video_embeds)
                    #    if self.local_rank == 0:
                            # Flatten combiend_text and combined_videos
                            # (world_size, batch_size, embed_dim) -> (world_size * batch_size, embed_dim)
                    #        combined_text = combined_text.view(-1, combined_text.shape[-1])
                    #        combined_videos = combined_videos.view(-1, combined_videos.shape[-1])
                    #        all_output = sim_matrix(combined_text, combined_videos)
                    #        loss = self.loss(all_output)
                    #        loss.backward()
                            #torch.save(1, f"/mnt/datasets_mnt/debug/loss_success_{random_int}.pt")
                    #        for param in self.model.parameters():
                    #            if param.requires_grad:
                    #                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                    if self.global_loss:
                        #import pdb; pdb.set_trace()
                        all_text_embeds = [torch.zeros_like(text_embeds) for _ in range(torch.distributed.get_world_size())]
                        all_video_embeds = [torch.zeros_like(video_embeds) for _ in range(torch.distributed.get_world_size())]

                        torch.distributed.all_gather(all_text_embeds, text_embeds)
                        torch.distributed.all_gather(all_video_embeds, video_embeds)

                        loss = self.loss(text_embeds, video_embeds, all_text_embeds, all_video_embeds, self.temperature)
                        #print("Text embeds:", text_embeds.shape)
                        #print("Number of text embeds:", len(all_text_embeds))
                    else:
                        output = sim_matrix(text_embeds, video_embeds, self.temperature)
                        #os.system('nvidia-smi') # ensure using all gpus
                        #print("Output shape:", output.shape)
                        loss = self.loss(output)
                    import pdb; pdb.set_trace()
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        if 'weight' or 'bias' in name:
                            if param.grad is not None:
                                count_dict[name] = 1
                    import pdb; pdb.set_trace()
                    print(count_dict)
                    #os.system('nvidia-smi') # see memory usage
                    self.optimizer.step()

                    # Cap temperature
                    if self.config["loss"]["args"]["temperature"]:
                        self.temperature.data = torch.clamp(self.temperature.data, 0.0001, 4.6052) # 0.0001 to log(100)

                    detached_loss = loss.detach().item()

                    if self.writer is not None:
                        if self.wandb:
                            if self.global_rank == 0:
                                self.writer({f'loss_train_{dl_idx}': detached_loss})
                        else:    
                            self.writer.log_scalar(f'loss_train_{dl_idx}', detached_loss)

                    total_loss[dl_idx] += detached_loss

                    progress.set_postfix({"dl": dl_idx, "loss": detached_loss})

                    self.optimizer.zero_grad()
                # Step after one batch from each dataloader
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                if batch_idx == self.len_epoch:
                    break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for data in tqdm(dl, desc=f"Validating dl{dl_idx}"):
                    meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)

                    # Note that if the batch is not scattered among all the GPUs, `DataParallel` will fail because
                    # the model's mandatory argument `data` will not be passed to some of them.
                    # It can happen with the last batch of the dataset, depending on its size.
                    # It could be safely ignored during training but on validation/test we want accurate metrics.
                    # This avoids using `DataParallel` in this case, and supposes this batch fits in one GPU.
                    current_batch_size = data['video'].shape[0]
                    if isinstance(self.model, nn.DataParallel) and current_batch_size < (dl.batch_size or 1):
                        scattered_len = len(self.model.scatter([torch.empty(current_batch_size)], {},
                                                               self.model.device_ids)[0])
                        avoid_data_parallel = scattered_len < len(self.model.device_ids)
                    else:
                        avoid_data_parallel = False

                    if avoid_data_parallel:
                        text_embed, vid_embed = self.model.module(data, return_embeds=True)
                    else:
                        text_embed, vid_embed = self.model(data, return_embeds=True)

                    if self.global_loss:
                        all_text_embeds = [torch.zeros_like(text_embed) for _ in range(torch.distributed.get_world_size())]
                        all_video_embeds = [torch.zeros_like(vid_embed) for _ in range(torch.distributed.get_world_size())]

                        torch.distributed.all_gather(all_text_embeds, text_embed)
                        torch.distributed.all_gather(all_video_embeds, vid_embed)

                        loss = self.loss(text_embed, vid_embed, all_text_embeds, all_video_embeds, self.temperature)

                        all_losses = [torch.zeros_like(loss) for _ in range(torch.distributed.get_world_size())]
                        torch.distributed.all_gather(all_losses, loss)
                        for l in all_losses:
                            total_val_loss[dl_idx] += l.item()

                        for t_e, v_e in zip(all_text_embeds, all_video_embeds):
                            text_embed_arr[dl_idx].append(t_e.cpu())
                            vid_embed_arr[dl_idx].append(v_e.cpu())
                    else:        
                        text_embed_arr[dl_idx].append(text_embed.cpu())
                        vid_embed_arr[dl_idx].append(vid_embed.cpu())
                        sims_batch = sim_matrix(text_embed, vid_embed)
                        
                        loss = self.loss(sims_batch)
                        total_val_loss[dl_idx] += loss.item()
        if self.wandb and self.global_rank == 0:
            log_dict = {}
        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None:
                if self.wandb:
                    if self.global_rank == 0:
                        log_dict[f'loss_val_{dl_idx}'] = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                else:    
                    self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            text_embeds = torch.cat(text_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])
            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims)
                verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                        mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    if self.wandb:
                        if self.global_rank == 0:
                            log_dict.update(to_write)
                    else:
                        for key, val in to_write.items():
                            self.writer.log_scalar(key, val)

                if self.visualizer is not None:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    for dl_idx in range(len(self.valid_data_loader))}
        res_dict['nested_val_metrics'] = nested_metrics
        if self.global_rank == 0:
            in_acc1, in_acc5, in_acc20 = run_imagenetv2_eval(self.model)
            ucf_acc1, ucf_acc5, ucf_acc20 = run_ucf101_eval(self.model)
            log_dict.update({'imagenetv2_acc1': in_acc1, 'imagenetv2_acc5': in_acc5, 'imagenetv2_acc20': in_acc20})
            log_dict.update({'ucf101_acc1': ucf_acc1, 'ucf101_acc5': ucf_acc5, 'ucf101_acc20': ucf_acc20})
            self.writer(log_dict)

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f" MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res

# Modified from https://github.com/Lightning-AI/lightning/blob/ab59f308b18622421edc67048d3b9fbfde96a9f4/src/pytorch_lightning/utilities/distributed.py#L143
class AllGatherGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor,
        group = torch.distributed.group.WORLD,
    ) -> Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor
    
    @staticmethod
    def backward(ctx, *grad_output) -> (Tensor, None):
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None