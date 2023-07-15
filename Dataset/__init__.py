"""
Dataset setup and loaders
This file including the different datasets processing pipelines
"""

import Dataset.Multi_data
from torch.utils.data import DataLoader
from Transform.L_transforms import data_transform_pipline

shuffle_tag = True


def setup_loaders(args):

    if args.dataset == "MultiData":
        LR_HR_set = Multi_data.TrainLR_HR(data_transform_pipline)
        LR_LR_set = Multi_data.TrainLR_LR()
        val_set = Multi_data.Validation_loader()
        LR_HR_train = DataLoader(LR_HR_set, batch_size=args.hr_batch_size, num_workers=args.num_workers,
                                 shuffle=shuffle_tag)
        LR_LR_train = DataLoader(LR_LR_set, batch_size=args.lr_batch_size, num_workers=args.num_workers,
                                 shuffle=shuffle_tag)
        val_loader = DataLoader(val_set, batch_size=args.hr_batch_size, num_workers=args.num_workers,
                                shuffle=shuffle_tag)
        return LR_HR_train, LR_LR_train, val_loader

    if args.dataset == "MultiData_wTC":
        LR_HR_set = Multi_data.TrainLR_HR(data_transform_pipline)
        LR_LR_set = Multi_data.TrainLR_LR_wTC()
        val_set = Multi_data.Validation_loader()
        LR_HR_train = DataLoader(LR_HR_set, batch_size=args.hr_batch_size, num_workers=args.num_workers,
                                 shuffle=shuffle_tag,pin_memory=False)
        LR_LR_train = DataLoader(LR_LR_set, batch_size=args.lr_batch_size, num_workers=args.num_workers,
                                 shuffle=shuffle_tag, drop_last=True,pin_memory=False)
        val_loader = DataLoader(val_set, batch_size=args.hr_batch_size, num_workers=args.num_workers,
                                shuffle=shuffle_tag,pin_memory=False)

        return LR_HR_train, LR_LR_train, val_loader
