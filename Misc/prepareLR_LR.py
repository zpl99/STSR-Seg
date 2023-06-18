# coding:utf-8

import os
import skimage.io
import numpy as np
from glob import glob
import tqdm
import csv
import argparse


def postProcessLrFiles(dataPath, T):
    path_d0 = os.listdir(dataPath)
    pbar = tqdm.tqdm(total=len(path_d0))
    with open('lrFiles.csv', 'w', encoding="UTF8", newline='') as file:
        writer = csv.writer(file)
        for i in path_d0:
            path_d1 = os.listdir(os.path.join(dataPath, i))
            for ii in path_d1:
                path_d2 = glob(os.path.join(dataPath, i, ii, "*s2*"))
                for imagePath in path_d2:
                    try:
                        labelPath = imagePath.replace("s2", "dynamicWorld")
                        label = skimage.io.imread(labelPath)
                        label = np.where(label >= T, 1, 0)
                        number = np.count_nonzero(label)
                        writer.writerow([imagePath, labelPath, number])
                    except:
                        continue
            pbar.update()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", type=str, default="/mnt/data/lzp/temp_lr")
    parser.add_argument("--T", type=float, default=0.5)
    args = parser.parse_args()
    args.inference = False
    return args


if __name__ == '__main__':
    args = get_args()
    postProcessLrFiles(args.dataPath, args.T)

