import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from Loss import loss
import numpy as np


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


def get_uncertain_point_coords_on_grid(mask):
    R, _, H, W = mask.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    # num_points = min(H * W, num_points)
    # point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_indices = torch.where(mask.view(R, H * W) != 0)[1]
    num_points = point_indices.shape[0]

    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=mask.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_coords


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

class Multi_data_train_framework(nn.Module):
    def __init__(self, sr, ss):
        super(Multi_data_train_framework, self).__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.BCE_loss_not_reduction = nn.BCEWithLogitsLoss()
        self.focalTloss = loss.FocalTverskyLoss(ALPHA=0.4, BETA=0.6, GAMMA=0.5)
        self.lr_loss = loss.SRLoss(torch.tensor(0.45), torch.tensor(0.097), reduction=True)
        self.sr = sr()
        self.ss = ss()
        self.avg_pool = nn.AvgPool2d(4, 4)  # H,W:256,256->64,64

    def forward(self, input, mode="train"):
        if mode == "train_hr":
            loss = {}
            lr_image = input["image"]
            hr_label = input["label"]
            feature = self.sr(lr_image)
            hr_pre= self.ss(feature)
            hr_loss = self.BCE_loss_not_reduction(hr_pre, hr_label) + 0.5 * self.focalTloss(hr_pre, hr_label)

            loss.update(
                {
                    "hr_loss": hr_loss,
                }
            )
            return loss, hr_pre
        elif mode=="train_lr":
            loss = {}
            lr_image = input["image"]
            lr_label = input["label"]
            hr_feature = self.sr(lr_image)
            hr_pre = self.ss(hr_feature)
            lr_pre = self.avg_pool(hr_pre)

            lr_bce_loss = self.BCE_loss_not_reduction(lr_pre, lr_label) + 0.5 * self.focalTloss(lr_pre, lr_label)
            lr_sr_loss = self.lr_loss(lr_pre, lr_label)
            loss.update(
                {
                    "lr_bce_loss": 1 * lr_bce_loss,
                    "lr_sr_loss": 0.025 * lr_sr_loss
                }
            )
            return loss

        elif mode == "inference":
            lr_image = input["image"]
            hr_feature = self.sr(lr_image)
            hr_pre = self.ss(hr_feature)
            return hr_pre
        elif mode == "val":
            lr_image = input["image"]
            hr_label = input["label"]
            hr_feature = self.sr(lr_image)
            hr_pre= self.ss(hr_feature)
            loss = self.BCE_loss_not_reduction(hr_pre, hr_label)
            return {"loss": loss}, hr_pre, hr_label
        else:
            raise ValueError
