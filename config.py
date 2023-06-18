##############################################################################
# Config
# Config is used to set dataset path for training and testing
##############################

from __future__ import absolute_import # 绝对引用
from __future__ import division # 精确除法 3/4=0.75 而不是0
from __future__ import print_function # 将python 2.0 的print替换成python 3.0的
from __future__ import unicode_literals

import torch

from Utils.attr_dict import AttrDict

__C = AttrDict()
cfg = __C

__C.DATASET = AttrDict()
__C.DATASET.VAL = "/mnt/data/lzp/temp_train_val/valData"
__C.DATASET.LR_HR = "/mnt/data/lzp/temp_train_val/trainData"
__C.DATASET.LR_LR = "Misc/lrFiles.csv"
__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = torch.nn.BatchNorm2d
def assert_and_infer_cfg(args, make_immutable=True, train_mode=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if args.dataset=="Inria-0.3":
        __C.DATASET.Inria="/media/dell/shihaoze/lzp/dataset/Inria-0.3"
    elif args.dataset=="Inria-0.6":
        __C.DATASET.Inria = "/media/dell/shihaoze/lzp/dataset/Inria-0.6"
    elif args.dataset=="Inria-0.9":
        __C.DATASET.Inria = "/media/dell/shihaoze/lzp/dataset/Inria-0.9"
    elif args.dataset=="Inria-1.2":
        __C.DATASET.Inria = "/media/dell/shihaoze/lzp/dataset/Inria-1.2"
    else:
        assert False, "please check dataset name further!"
    if hasattr(args, 'syncbn') and args.syncbn:
        if args.apex:
            import apex
            __C.MODEL.BN = 'apex-syncnorm'
            __C.MODEL.BNFUNC = apex.parallel.SyncBatchNorm
        else:
            raise Exception('No Support for SyncBN without Apex')
    else:
        __C.MODEL.BNFUNC = torch.nn.BatchNorm2d
        print('Using regular batch norm')

    if not train_mode:
        cfg.immutable(True)
        return
    if make_immutable:
        cfg.immutable(True)