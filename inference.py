from Utils import process_tools
import glob
import os
import numpy as np
import Framework
from Transform import L_transforms
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import skimage.io
import math
import shutil
import skimage.transform
import argparse
from Dataset import Multi_data
import Nets


def get_args():
    parser = argparse.ArgumentParser(description='National building mapping')
    parser.add_argument("--s2Path", type=str, default="/mnt/data/lzp/STSRSeg/Inference_data_path/S2")
    parser.add_argument("--luccPath", type=str,
                        default="/mnt/data/lzp/STSRSeg/Inference_data_path/DynamicWorld")
    parser.add_argument("--checkpointPath", type=str,
                        default="/mnt/data/lzp/STSRSeg/MultiDataEDSRUnet-model-400.ckpt")
    parser.add_argument("--framework", type=str, default="inference")
    parser.add_argument("--sr", type=str,
                        default="EDSR")
    parser.add_argument("--ss", type=str,
                        default="Unet")
    parser.add_argument("--desPath", type=str, default="/mnt/data/lzp/STSRSeg/Pre/")
    parser.add_argument("--tempPath", type=str, default="/mnt/data/lzp/STSRSeg/Temp")
    args = parser.parse_args()
    return args


def load_checkpoint(net, model_path):
    checkpoint = torch.load(model_path)
    net_state = checkpoint["state_dict"]
    net.load_state_dict(net_state)
    return net


def parse_output(outs, image_names, save_path=None):
    if save_path != None:
        save_path = save_path
    else:
        save_path = "predictions"
    os.makedirs(save_path, exist_ok=True)
    for i in range(outs.shape[0]):
        out = outs[i]
        out = torch.from_numpy(out)
        np.save(f"{save_path}/{image_names[i]}", out[0])  # out[0] -> 1,256,256 to 256,256


def inference(dataloader, net, save_path=None):
    net.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            out = net(batch)
            out = torch.sigmoid(out)
            out = out.detach().cpu().numpy()
            out = np.where(out > 0.5, 1, 0)
            parse_output(out, batch["image_name"], save_path)
    return


def make_predictions(folder_path, net, save_path):
    dataset = Multi_data.LRDataset_for_inference(L_transforms.data_transform_pipline, folder_path)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=16,
                             pin_memory=True, shuffle=True)

    inference(data_loader, net, save_path=save_path)


if __name__ == "__main__":
    args = get_args()
    print(args)
    file_path = args.s2Path
    lucc_paths = args.luccPath
    model_path = args.checkpointPath

    patches = glob.glob(os.path.join(file_path, "*.tif"))
    pbar = tqdm(total=len(patches), desc="Output prediction...")
    net = Framework.setup_framework(args)
    net = Nets.wrap_network_in_dataparallel(net, False)
    model = torch.load(model_path)
    trained_dict = {k: v for k, v in model.items() if k in net.state_dict()}
    net.load_state_dict(trained_dict)

    for patch in patches:
        base_name = os.path.basename(patch)
        base_name_wo_suffix = base_name.split(".")[0]
        # sacrifice storage space for more memory (this is really weird, but we have limited server memory)
        if not os.path.exists(args.tempPath + base_name_wo_suffix + "_subpatch_folder"):
            os.makedirs(args.tempPath + base_name_wo_suffix + "_subpatch_folder")
        if not os.path.exists(args.tempPath + base_name_wo_suffix + "_outpredictions_folder"):
            os.makedirs(args.tempPath + base_name_wo_suffix + "_outpredictions_folder")
        # split the image to small patches and get the index according to the positions
        indices = process_tools.crop_one_image_with_padding(patch,
                                                            args.tempPath + base_name_wo_suffix + "_subpatch_folder",
                                                            64)
        make_predictions(args.tempPath + base_name_wo_suffix + "_subpatch_folder", net,
                         args.tempPath + base_name_wo_suffix + "_outpredictions_folder")

        image = skimage.io.imread(patch)
        image_h, image_w = image.shape[0], image.shape[1]
        del image  # due to limited server memory
        # composite parches into one
        composited_image = process_tools.composite_patch_to_one_image(
            args.tempPath + base_name_wo_suffix + "_outpredictions_folder", indices=indices)
        # use lucc data to do masking
        lucc_path = os.path.join(lucc_paths, base_name)
        lucc = skimage.io.imread(lucc_path)
        lucc = np.nan_to_num(lucc)
        lucc = skimage.transform.resize(lucc, composited_image.shape)
        lucc = np.where(lucc >= 0.2, 1, 0)

        process_tools.project_image(patch,
                                    composited_image,
                                    os.path.join(args.desPath, f"{base_name_wo_suffix}_pre.tif"),
                                    image_h * 4, image_w * 4, 1, lucc=lucc)
        # clean temporary files
        shutil.rmtree(args.tempPath + base_name_wo_suffix + "_subpatch_folder")
        shutil.rmtree(args.tempPath + base_name_wo_suffix + "_outpredictions_folder")
        pbar.update()
        del image_h, image_w, composited_image, base_name_wo_suffix  # due to limited server memory
