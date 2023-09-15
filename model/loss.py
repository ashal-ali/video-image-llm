import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

def sim_matrix(a, b, temperature = 1.0, eps=1e-8):
    """
    added eps for numerical stability
    """
    scale = torch.exp(temperature)
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = scale * torch.mm(a_norm, b_norm.transpose(0, 1))
    print(sim_mt)
#    import pdb; pdb.set_trace()
    return sim_mt

class GlobalNormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=True):
        super().__init__()
        self.rank = dist.get_rank()
    
    def get_idxs(self, b): # Get idxs for local embeds with batch_size = b
        #import pdb; pdb.set_trace()
        j = torch.arange(b)
        i = self.rank * b + j
        #idxs = torch.stack([i, j], dim=1)
        idxs_flipped = torch.stack([j, i], dim=1)
        return idxs_flipped

    def forward(self, local_videos, local_texts, global_videos, global_texts, temperature):
        # unwrap global from list to torch tensor
        global_videos = torch.cat(global_videos)
        global_texts = torch.cat(global_texts)

        #import pdb; pdb.set_trace()

        # use subset of global videos and texts
        bs = local_videos.shape[0]
        vid_subset = global_videos#[bs*self.rank:bs*(self.rank+1)]
        text_subset = global_texts#[bs*self.rank:bs*(self.rank+1)]
        #print(f"Grabbing subset of global videos and texts from {bs*self.rank} to {bs*(self.rank+1)} for rank {self.rank}")

        # compute similarity matrix
        #print(f"Computing similarity between: {local_videos.shape} and {text_subset.shape}")
        vid_sim = sim_matrix(local_videos, text_subset, temperature)
        text_sim = sim_matrix(local_texts, vid_subset, temperature)

        #print("Video similarity matrix shape:", vid_sim.shape)
        #print("Text similarity matrix shape:", text_sim.shape)

        # compute loss
        i_logsm = F.log_softmax(vid_sim, dim=1)
        j_logsm = F.log_softmax(text_sim, dim=1)

        # sum over positives (get idxs)
        idxs = self.get_idxs(bs) # TODO: Cache at beginning? 
        #print(f"idxs: {idxs}")
        #print("i_logsm:", i_logsm.shape)
        #print("j_logsm:", j_logsm.shape)
        #import pdb; pdb.set_trace()
        vals_i = i_logsm[idxs[:, 0], idxs[:, 1]]
        loss_i = vals_i.sum() / len(idxs)

        vals_j = j_logsm[idxs[:, 0], idxs[:, 1]]
        loss_j = vals_j.sum() / len(idxs)

        return - loss_i - loss_j
    

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def nll_loss(output, target):
    return F.nll_loss(output, target)


if __name__ == "__main__":
    import torch

    random_sims = (torch.rand([10, 8]) * 2) - 1
    loss = NormSoftmaxLoss()
    loss(random_sims)


"""
Adding code from Huggingface CLIP implementation, convert to this format in the future:

        Calling the loss:        
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)
        
        loss function: 
        # contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor, dim: int) -> torch.Tensor:
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0
"""