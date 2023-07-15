# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from Loss import loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


class Multi_data_train_framework_wTC(nn.Module):

    def __init__(self, sr, ss, dim=128, K=16, m=0.999, T=0.07, out_size=(256, 256)):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(Multi_data_train_framework_wTC, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.out_size = out_size

        """
        loss definitions
        """
        self.info_nce = nn.CrossEntropyLoss()
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.BCE_loss_not_reduction = nn.BCEWithLogitsLoss()
        self.focalTloss = loss.FocalTverskyLoss(ALPHA=0.4, BETA=0.6, GAMMA=0.5)
        self.lr_loss = loss.SRLoss(torch.tensor(0.45), torch.tensor(0.097), reduction=True)
        self.sr = sr()
        self.ss = ss()
        model = nn.Sequential(self.sr, self.ss)
        self.encoder_q = model
        self.encoder_k = copy.deepcopy(model)
        self.avg_pool = nn.AvgPool2d(4, 4)  # H,W:256,256->64,64

        print("model creat!")
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        print("momentum model creat!")
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, input, mode):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        if mode == "train_hr":
            loss = {}
            lr_image = input["image"]
            hr_label = input["label"]
            hr_pre, _ = self.encoder_q(lr_image)
            hr_loss = self.BCE_loss_not_reduction(hr_pre, hr_label) + 0.5 * self.focalTloss(hr_pre, hr_label)

            loss.update(
                {
                    "hr_loss": hr_loss,
                }
            )
            return loss, hr_pre
        elif mode == "train_lr":
            loss = {}
            im_q = input["image"]
            im_k = input["image_tc"]
            lr_label = input["label"]

            hr_pre, q = self.encoder_q(im_q)
            lr_pre = self.avg_pool(hr_pre)
            q = nn.functional.normalize(q, dim=1)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                k = self.encoder_k(im_k)[-1]  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            tc_loss = self.info_nce(logits, labels)
            lr_bce_loss = self.BCE_loss_not_reduction(lr_pre, lr_label) + 0.5 * self.focalTloss(lr_pre, lr_label)
            lr_sr_loss = self.lr_loss(lr_pre, lr_label)
            loss.update(
                {
                    "lr_bce_loss": 1 * lr_bce_loss,
                    "lr_sr_loss": 0.025 * lr_sr_loss,
                    "tc_loss": 0.005 * tc_loss
                }
            )
            return loss
        elif mode == "val":
            lr_image = input["image"]
            hr_label = input["label"]
            hr_pre, _ = self.encoder_q(lr_image)
            loss = self.BCE_loss_not_reduction(hr_pre, hr_label)
            return {"loss": loss}, hr_pre, hr_label


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


