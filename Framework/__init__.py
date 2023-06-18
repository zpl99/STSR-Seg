from Framework import multi_data_source, multi_data_source_wTC, inference_framework
import torch
import numpy as np
import Nets
from Nets.ESPC import ESPC
from Nets.edsr import EDSR
from Nets.unet import U_Net, U_Net_wTC
from Nets.RRDBNet import RRDBNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_framework(args):
    if args.framework == "lr_lr_and_lr_hr":
        if args.sr == "ESPC":
            sr = ESPC
        elif args.sr == "RRDB":
            sr = RRDBNet
        elif args.sr == "EDSR":
            sr = EDSR
        else:
            raise ValueError

        if args.ss == "Unet":
            ss = U_Net
        else:
            raise ValueError
        framework = multi_data_source.Multi_data_train_framework(sr, ss)
        return framework
    if args.framework == "lr_lr_and_lr_hr_wTC":
        if args.sr == "ESPC":
            sr = ESPC
        elif args.sr == "RRDB":
            sr = RRDBNet
        elif args.sr == "EDSR":
            sr = EDSR
        else:
            raise ValueError
        if args.ss == "Unet":
            ss = U_Net_wTC
        else:
            raise ValueError
        framework = multi_data_source_wTC.Multi_data_train_framework_wTC(sr, ss)
        return framework

    if args.framework == "inference":
        if args.sr == "ESPC":
            sr = ESPC
        elif args.sr == "RRDB":
            sr = RRDBNet
        elif args.sr == "EDSR":
            sr = EDSR
        else:
            raise ValueError
        if args.ss == "Unet":
            ss = U_Net
        else:
            raise ValueError
        framework = inference_framework.inference_framework(sr, ss)
        return framework

if __name__ == "__main__":
    model = torch.load(r"/mnt/data/lzp/STSRSeg/MultiDataEDSRUnet-model-15.ckpt")
    sr = EDSR
    ss = U_Net
    framework = inference_framework.inference_framework(sr,ss)
    framework = Nets.wrap_network_in_dataparallel(framework, False)
    pretrained_dict = {k: v for k, v in model.items() if k in framework.state_dict()}

    framework.load_state_dict(pretrained_dict)