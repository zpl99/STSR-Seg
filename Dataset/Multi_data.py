import os
import os.path as osp
import numpy as np
from torch.utils import data
from config import cfg
import skimage.io
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
from glob import glob
import random

# the path of the data
LR_HR = cfg.DATASET.LR_HR
LR_LR = cfg.DATASET.LR_LR
VAL = cfg.DATASET.VAL

colorjitter = transforms.RandomApply([
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
], p=0.2)
normalize = transforms.Normalize(mean=[0.14172366, 0.12568618, 0.12004076, 0.1804051],
                                 std=[0.03957363, 0.04393258, 0.0611819, 0.0827849])


def center_crop(img, out_size):
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = int((image_height - crop_height + 1) * 0.5)
    crop_left = int((image_width - crop_width + 1) * 0.5)
    return img[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]


def make_dataset_for_inference(file_path):
    img_tokens = os.listdir(file_path)
    img_list = []
    for i in img_tokens:
        img_list.append(os.path.join(file_path, i))
    return img_list

def makeDataset_with_hr_label():
    all_tokens = []
    image_path = osp.join(LR_HR, "image")
    label_path = osp.join(LR_HR, "label")
    img_tokens = os.listdir(image_path)
    img_tokens.sort()

    label_tokens = os.listdir(label_path)
    label_tokens.sort()
    for img_token, label_token in zip(img_tokens, label_tokens):
        token = (osp.join(image_path, img_token), osp.join(label_path, label_token))
        all_tokens.append(token)
    print(f"LR and HR combination contains {len(all_tokens)} samples")
    return all_tokens

def makeDataset_with_lr_label(percentage=0.3):
    data = np.genfromtxt(LR_LR, delimiter=' ', dtype=str)
    data_tokens = []
    for per_data in data:
        image_path, label_path, count = per_data.split(",")
        if int(count) > 4096 * percentage:  # 4096=64*64
            data_tokens.append([image_path, label_path])

    print(
        f"LR and LR combination contains {len(data_tokens)} samples (only samples with a pixel share of {percentage} or more in the built-up area were included)")
    return data_tokens

def makeDataset_with_lr_label_wTC(percentage=0.3):

    data = np.genfromtxt("./Misc/lrFiles.csv", delimiter=' ', dtype=str)
    data_tokens = []
    for per_data in data:
        image_path, label_path, count = per_data.split(",")
        if int(count) > 4096 * percentage:  # 4096=64*64
            dir = os.path.dirname(image_path)
            all_files = glob(os.path.join(dir, "*s2.tif"))
            all_files.remove(image_path)
            tc_path = random.sample(all_files, 1)[0]
            data_tokens.append([image_path, tc_path, label_path])

    print(f"LR and LR combination contains {len(data_tokens)} samples (only samples with a pixel share of {percentage} or more in the built-up area were included)")
    return data_tokens
def makeValfolder():
    all_tokens = []
    image_path = osp.join(cfg.DATASET.VAL, "image")
    label_path = osp.join(cfg.DATASET.VAL, "label")

    img_tokens = os.listdir(image_path)
    img_tokens.sort()

    label_tokens = os.listdir(label_path)
    label_tokens.sort()

    for img_token, label_token in zip(img_tokens, label_tokens):
        token = (osp.join(image_path, img_token), osp.join(label_path, label_token))
        all_tokens.append(token)
    print(f"validation set contains {len(all_tokens)} samples")
    return all_tokens


class TrainLR_HR(data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.data_tokens = makeDataset_with_hr_label()
        assert len(self.data_tokens), "the data is empty!Please check the root"

    def __getitem__(self, index):
        image_path, label_path = self.data_tokens[index]
        image, label = skimage.io.imread(image_path), skimage.io.imread(label_path)

        label = np.where(label != 0, 1, 0)
        image, label = self.transform(image, label, ["roate", "vflipAndhflip", "color_jittering"])
        result = {
            "image": image,
            "label": label
        }
        return result

    def __len__(self):
        return len(self.data_tokens)


class Validation_loader(data.Dataset):
    def __init__(self):
        self.data_tokens = makeValfolder()
        assert len(self.data_tokens), "the data is empty!Please check the root"

    def __getitem__(self, index):
        lr_image_path, hr_label_path = self.data_tokens[index]
        lr_image = skimage.io.imread(lr_image_path)
        lr_image = torch.from_numpy(lr_image.transpose((2, 0, 1)))
        lr_image = lr_image.to(torch.float32)
        lr_image = TF.normalize(lr_image, mean=[0.14172366, 0.12568618, 0.12004076, 0.1804051],
                                std=[0.03957363, 0.04393258, 0.0611819, 0.0827849])

        hr_label = skimage.io.imread(hr_label_path)
        hr_label[hr_label != 0] = 1
        hr_label = torch.from_numpy(hr_label)
        hr_label = hr_label.unsqueeze(0)
        hr_label = hr_label.to(torch.float32)

        result = {
            "image": lr_image,
            "label": hr_label
        }
        return result

    def __len__(self):
        return len(self.data_tokens)

class TrainLR_LR_wTC(data.Dataset):
    def __init__(self):
        self.data_tokens = makeDataset_with_lr_label_wTC()
        assert len(self.data_tokens), "the data is empty!Please check the root"

    def __getitem__(self, index):
        token = self.data_tokens[index]
        image_path, tc_path, label_path = token

        image = skimage.io.imread(image_path)
        image = center_crop(image, (64, 64))
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.to(torch.float32)

        image_tc = skimage.io.imread(tc_path)
        image_tc = center_crop(image_tc, (64, 64))
        image_tc = image_tc.transpose((2, 0, 1))
        image_tc = torch.from_numpy(image_tc)
        image_tc = image_tc.to(torch.float32)


        label = skimage.io.imread(label_path)
        label = np.where(label > 0.5, 1, 0)
        label = center_crop(label, (64, 64))
        label = torch.from_numpy(label)  # [H,W]
        label = label.unsqueeze(0)
        label = label.to(torch.float32)

        image_tc_rgb = image_tc[:3, :, :]
        image_tc_nir = image_tc[3, :, :].unsqueeze(0)
        image_tc_rgb = colorjitter(image_tc_rgb)
        image_tc = torch.cat([image_tc_rgb, image_tc_nir], dim=0)
        image_tc = normalize(image_tc)
        
        image_rgb = image[:3, :, :]
        image_nir = image[3, :, :].unsqueeze(0)
        image_rgb = colorjitter(image_rgb)
        image = torch.cat([image_rgb, image_nir], dim=0)
        image = normalize(image)

        result = {
            "image": image,
            "image_tc":image_tc,
            "label": label
        }
        return result

    def __len__(self):
        return len(self.data_tokens)
class TrainLR_LR(data.Dataset):
    def __init__(self):
        self.data_tokens = makeDataset_with_lr_label()
        assert len(self.data_tokens), "the data is empty!Please check the root"

    def __getitem__(self, index):
        token = self.data_tokens[index]
        image_path, label_path = token

        image = skimage.io.imread(image_path)
        image = center_crop(image, (64, 64))
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.to(torch.float32)

        label = skimage.io.imread(label_path)
        label = np.where(label > 0.5, 1, 0)
        label = center_crop(label, (64, 64))
        label = torch.from_numpy(label)  # [H,W]
        label = label.unsqueeze(0)
        label = label.to(torch.float32)

        image_rgb = image[:3, :, :]
        image_nir = image[3, :, :].unsqueeze(0)
        image_rgb = colorjitter(image_rgb)
        image = torch.cat([image_rgb, image_nir], dim=0)
        image = normalize(image)

        result = {
            "image": image,
            "label": label
        }
        return result

    def __len__(self):
        return len(self.data_tokens)
class LRDataset_for_inference(data.Dataset):
    def __init__(self, transform, file_path):
        self.transform = transform
        self.data_tokens = make_dataset_for_inference(file_path)
        assert len(self.data_tokens), "the data is empty!Please check the root"

    def __getitem__(self, index):
        token = self.data_tokens[index]
        image_path = token

        image = skimage.io.imread(image_path)
        label = np.ones([4, 256, 256])

        image_name = osp.splitext(osp.basename(image_path))[0]
        image, label = self.transform(image, label)
        result = {
            "image": image,
            "label": label,
            "image_name": image_name
        }
        return result

    def __len__(self):
        return len(self.data_tokens)