import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

_euler_num = 2.718281828  # euler number
_pi = 3.14159265  # pi
_ln_2_pi = 1.837877  # ln(2 * pi)
_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        device = predict.device
        target = target.contiguous().view(target.shape[0], -1)
        target_gpu = target.clone().cuda(device=device)
        valid_mask_gpu = valid_mask.clone().cuda(device=device)
        valid_mask_gpu = valid_mask_gpu.contiguous().view(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(predict, target_gpu) * valid_mask_gpu, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target_gpu.pow(self.p)) * valid_mask_gpu, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        # self.weight = weight

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )


class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False, ohem=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.fp16 = False
        self.ohem = ohem
        self.ohem_loss = OhemCrossEntropy2dTensor(self.ignore_index).cuda()

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()  # c
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def onehot2label(self, target):
        # a bug here
        label = torch.argmax(target[:, :-1, :, :], dim=1).long()
        label[target[:, -1, :, :]] = self.ignore_index
        return label

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            if self.ohem:
                return self.ohem_loss(inputs, self.onehot2label(target))
            border_weights = 1 / border_weights
            target[target > 1] = 1
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                          (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        # add ohem loss for the final stage
        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH) and self.ohem:
            return self.ohem_loss(inputs, self.onehot2label(target[:, :-1, :, :]))
        if self.fp16:
            weights = target[:, :-1, :, :].sum(1).half()
        else:
            weights = target[:, :-1, :, :].sum(1).float()  # nxhxw(0-3)
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1
        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights, mask=ignore_mask[i])

        return loss


class OhemWithAux(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000, aux_weight=0.4):
        super(OhemWithAux, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.aux_weight = aux_weight
        self.main_loss = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)
        self.aux_loss = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, pred, target):
        x_main, x_aux = pred
        return self.main_loss(x_main, target) + self.aux_weight * self.aux_loss(x_aux, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)







def kld_gauss(u1, s1, u2, s2):
    # general KL two Gaussians
    # u2, s2 often N(0,1)
    # https://stats.stackexchange.com/questions/7440/ +
    # kl-divergence-between-two-univariate-gaussians
    # log(s2/s1) + [( s1^2 + (u1-u2)^2 ) / 2*s2^2] - 0.5
    v1 = s1 * s1
    v2 = s2 * s2
    a = torch.log((s2 / (s1 + 0.000001)))
    num = v1 + (u1 - u2) ** 2
    den = 2 * v2
    b = num / den
    return a + b - 0.5


def wasserstein(mu_1, sigma_1, mu_2, sigma_2):
    p1 = torch.sum(torch.pow((mu_1 - mu_2), 2), dim=1)
    # print("sigma", sigma_1.shape, sigma_1)
    # print("sigma", sigma_2.shape, sigma_2)
    p2 = torch.sum(torch.pow(torch.pow(torch.abs(sigma_1), 1 / 2) - torch.pow(torch.abs(sigma_2), 1 / 2), 2), 1)
    return torch.mean(p1 + p2)


def univar_continue_KL_divergence(p, q):
    # p is target distribution

    return torch.log(q[1] / p[1] + 0.000001) + (p[1] ** 2 + (p[0] - q[0]) ** 2) / (2 * q[1] ** 2) - 0.5


class SRLoss(nn.Module):
    def __init__(self, nlcd_means, nlcd_vars, sigmoid_scale = 1, reduction=True):
        super(SRLoss, self).__init__()
        self.nlcd_means = nlcd_means
        self.nlcd_vars = nlcd_vars
        self.reduction = reduction
        self.sigmoid_scale = sigmoid_scale
    def ddist(self, prediction, c_interval_center, c_interval_radius):
        return F.relu(torch.abs(prediction.sub(c_interval_center)).sub(c_interval_radius))

    def forward(self, y_out, y_true, from_logits=True):
        # preprocess
        if from_logits:
            # softmax = nn.Softmax(dim=1)
            sigmoid = nn.Sigmoid()
            y_pred = sigmoid(self.sigmoid_scale*y_out)
        else:
            y_pred = y_out
        mask_size = torch.unsqueeze(torch.sum(y_true, dim=(1, 2, 3)) + 10, 1)
        c_mask = y_true  # shape Bx1xHxW
        c_mask_size = torch.sum(c_mask, dim=(2, 3)) + 0.000001  # shape Bx1 (smoothing in case of 0)
        c_interval_center = self.nlcd_means  # shape hr_classes = C
        c_interval_radius = self.nlcd_vars  # shape hr_classes
        masked_probs = y_pred * c_mask
        # BxCxHxW * BxHxW --> BxCxHxW
        # Mean mean of predicted distribution
        mean = (
                torch.sum(masked_probs, dim=(2, 3)) / c_mask_size
        )  # (B,hr_classes) / (B,1) --> shape Bxhr_classes
        # Mean var of predicted distribution
        var = torch.sum(masked_probs * (1.0 - masked_probs), dim=(2, 3)) / (
                c_mask_size * c_mask_size
        )  # (B,hr_classes) / (B,1) --> shape Bxhr_classes
        c_loss = kld_gauss(mean, var, c_interval_center, c_interval_radius)

        loss = c_loss * (c_mask_size / mask_size)
        if self.reduction:
            return torch.sum(loss, dim=1).mean()
        else:
            return torch.sum(loss, dim=1)


class FocalTverskyLoss(nn.Module):
    """
    https://github.com/anwai98/Loss-Functions/blob/main/loss-function-library-keras-pytorch.ipynb
    """

    def __init__(self, ALPHA=0.3, BETA=0.7, GAMMA=2.0):
        super(FocalTverskyLoss, self).__init__()
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA

    def forward(self, inputs, targets, smooth=1e-6):
        alpha = self.ALPHA
        beta = self.BETA
        gamma = self.GAMMA
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


if __name__ == '__main__':
    loss = SRLoss(torch.tensor(0.89), torch.tensor(0.17), reduction=True)
    pre = torch.zeros((2, 1, 64, 64))
    gt = torch.ones((2, 1, 64, 64))
    l = loss(pre, gt)
    print(l)
