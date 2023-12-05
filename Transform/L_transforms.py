import os
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import ImageFilter

"""Data augmentation for semantic segmentation data"""


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def moco_transforms(image):
    normalize = transforms.Normalize(mean=[0.14172366, 0.12568618, 0.12004076, 0.1804051],
                                     std=[0.03957363, 0.04393258, 0.0611819, 0.0827849])
    resize_crop = transforms.RandomResizedCrop(64, scale=(0.2, 1.))  # allow 4 bands
    colorjitter = transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8)
    grayscale = transforms.RandomGrayscale(p=0.2)
    # gaussianblur = transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5)
    hfilp = transforms.RandomHorizontalFlip()

    # forward
    # the input is a 4-channel image
    image = resize_crop(image)

    rgb = image[:3, :, :]  # 3,H,W
    nir = image[3, :, :]  # H,W
    nir = nir.unsqueeze(0)  # 1,H,W
    rgb = colorjitter(rgb)
    rgb = grayscale(rgb)
    # rgb = gaussianblur(rgb)

    image = torch.cat((rgb, nir), dim=0)  # 4,H,W

    image = hfilp(image)
    image = normalize(image)

    return image


def roate(image, mask):
    """The image rotates randomly with a 50% probability"""
    if random.random() > 0.5:
        angle = random.randint(-20, 20)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
    return image, mask


def vflipAndhflip(image, mask):
    """vertical reverse"""
    if random.random() > 0.5:
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if random.random() <= 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
    return image, mask


def vflipAndhflip2(image, lr_mask, lr_gradient, hr_mask, hr_gradient):
    """vertical reverse"""
    if random.random() > 0.5:
        if random.random() > 0.5:
            image = TF.vflip(image)
            lr_mask = TF.vflip(lr_mask)
            lr_gradient = TF.vflip(lr_gradient)
            hr_gradient = TF.vflip(hr_gradient)
            hr_mask = TF.vflip(hr_mask)
        if random.random() <= 0.5:
            image = TF.hflip(image)
            lr_mask = TF.hflip(lr_mask)
            lr_gradient = TF.hflip(lr_gradient)
            hr_gradient = TF.hflip(hr_gradient)
            hr_mask = TF.hflip(hr_mask)
    return image, lr_mask, lr_gradient, hr_mask, hr_gradient


def data_transform_pipline(image, mask, pipline=None, size=None):
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    # print(image.shape)
    mask = mask / 1.0
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)

    image = image.to(torch.float32)
    mask = mask.to(torch.float32)

    if size is not None:
        image = TF.resize(image, size)
        mask = TF.resize(mask, size)
    if pipline is not None:
        for i in pipline:
            if i == "roate":
                image, mask = roate(image, mask)
            elif i == "vflipAndhflip":
                image, mask = vflipAndhflip(image, mask)
            elif i == "color_jittering":
                colorjitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
                image_rgb = image[:3, :, :]
                image_nir = image[3, :, :].unsqueeze(0)
                image_rgb = colorjitter(image_rgb)
                image = torch.cat([image_rgb, image_nir], dim=0)
            else:
                break

    image = TF.normalize(image, mean=[0.14172366, 0.12568618, 0.12004076, 0.1804051],
                         std=[0.03957363, 0.04393258, 0.0611819, 0.0827849])

    return image, mask


def data_transform_pipline_multi_data_framework(lr_image, lr_label, hr_label):
    # print(image.shape)
    hr_label = hr_label / 1.0
    hr_label = torch.from_numpy(hr_label)
    hr_label = hr_label.div(255)
    hr_label = hr_label.unsqueeze(0)

    lr_label = lr_label / 1.0
    lr_label = torch.from_numpy(lr_label)
    lr_label = lr_label.div(255)
    lr_label = lr_label.unsqueeze(0)

    # lr_image = lr_image.to(torch.float32).cuda()
    # lr_gradient = lr_gradient.to(torch.float32).cuda()
    # hr_gradient = hr_gradient.to(torch.float32).cuda()
    hr_label = hr_label.to(torch.float32)
    lr_label = lr_label.to(torch.float32)

    lr_image = TF.resize(lr_image, [64, 64])
    # lr_gradient = TF.resize(lr_gradient, [64, 64])
    # hr_gradient = TF.resize(hr_gradient, [256, 256])
    hr_label = TF.resize(hr_label, [256, 256])
    lr_label = TF.resize(lr_label, [64, 64])

    lr_image[lr_image != lr_image] = 0
    # lr_gradient[lr_gradient != lr_gradient] = 0
    # hr_gradient[hr_gradient != hr_gradient] = 0
    hr_label[hr_label != hr_label] = 0
    lr_label[lr_label != lr_label] = 0

    # lr_image, lr_label, lr_gradient, hr_label, hr_gradient = vflipAndhflip2(lr_image, lr_label, lr_gradient, hr_label,
    #                                                                         hr_gradient)
    colorjitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
    image_rgb = lr_image[:3, :, :]
    image_nir = lr_image[3, :, :].unsqueeze(0)
    image_rgb = colorjitter(image_rgb)
    image = torch.cat([image_rgb, image_nir], dim=0)

    lr_image = TF.normalize(image, mean=[0.14172366, 0.12568618, 0.12004076, 0.1804051],
                            std=[0.03957363, 0.04393258, 0.0611819, 0.0827849])

    return lr_image, lr_label, hr_label


def data_transform_pipline_robust_framework(image, mask, pipline=None, size=None):
    """
    Data enhancement of TP Positive is added.
    The returned data includes image,mask (used to supervise training) and image_positive(used for TP comparison learning).
    """
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    # print(image.shape)
    mask = mask / 1.0
    mask = torch.from_numpy(mask)
    mask = mask.div(255)
    mask = mask.unsqueeze(0)

    image = image.to(torch.float32)
    mask = mask.to(torch.float32)

    if size is not None:
        image = TF.resize(image, size)
        mask = TF.resize(mask, size)

    image_positive = moco_transforms(image)

    if pipline is not None:
        for i in pipline:
            if i == "roate":
                image, mask = roate(image, mask)
            elif i == "vflipAndhflip":
                image, mask = vflipAndhflip(image, mask)
            elif i == "color_jittering":
                colorjitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
                image_rgb = image[:3, :, :]
                image_nir = image[3, :, :].unsqueeze(0)
                image_rgb = colorjitter(image_rgb)
                image = torch.cat([image_rgb, image_nir], dim=0)
            else:
                break

    image = TF.normalize(image, mean=[0.14172366, 0.12568618, 0.12004076, 0.1804051],
                         std=[0.03957363, 0.04393258, 0.0611819, 0.0827849])

    return image, mask, image_positive


def data_transform_pipline_semi_supervised(image, mask, pipline=None, size=None):
    """
    Some changes were made because of the data format and other reasons
    """
    # image = torch.from_numpy(image)
    # print(image.shape)
    # mask = torch.from_numpy(mask)
    # mask = mask.div(255)
    # mask = mask.unsqueeze(0)

    image = image.to(torch.float32)
    mask = mask.to(torch.float32)

    if size is not None:
        image = TF.resize(image, size)
        mask = TF.resize(mask, size)
    if pipline is not None:
        for i in pipline:
            if i == "roate":
                image, mask = roate(image, mask)
            elif i == "vflipAndhflip":
                image, mask = vflipAndhflip(image, mask)
            elif i == "color_jittering":
                colorjitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
                image_rgb = image[:3, :, :]
                image_nir = image[3, :, :].unsqueeze(0)
                image_rgb = colorjitter(image_rgb)
                image = torch.cat([image_rgb, image_nir], dim=0)
            else:
                break

    image = TF.normalize(image, mean=[0.14172366, 0.12568618, 0.12004076, 0.1804051],
                         std=[0.03957363, 0.04393258, 0.0611819, 0.0827849])

    return image, mask


if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\dell\Desktop\Code\MyCDCode\data\gaofen_tiny\images\train\1_1.png")
    label = cv2.imread(r"C:\Users\dell\Desktop\Code\MyCDCode\data\gaofen_tiny\gt\train\1_1_label.png")
    image, mask = roate(image, label)
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()
