import os
from glob import glob
from skimage import io
import numpy as np
import rasterio
from affine import Affine
import skimage.io
import skimage.transform
from Utils.split import EMPatches
import re

patch_w, patch_h = 64, 64
em = EMPatches()


class SortNum(object):
    def __init__(self, lst):
        self.lst = lst

    @staticmethod
    def convert2int(s):
        try:
            return int(s)
        except ValueError:
            return s

    def str2int(self, v_str):
        return [self.convert2int(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

    def sort_num(self):
        return sorted(self.lst, key=self.str2int)


def crop_one_image_with_padding(single_image_path, save_folder_path, patch_size):
    image = io.imread(single_image_path)
    patches, indices = em.extract_patches(image, patchsize=patch_size, overlap=0.1)
    for i in range(len(patches)):
        patch = patches[i]
        io.imsave(rf"{save_folder_path}/{os.path.basename(single_image_path).split('.')[0]}_{i}.tif", patch)
    return indices


def composite_patch_to_one_image(patch_image_folder, indices):
    filename = glob(patch_image_folder + "/*.npy")
    filename = SortNum(filename).sort_num()
    patch_list = []
    for i in filename:
        pre = np.load(i)
        patch_list.append(pre)
    image = em.merge_patches(patch_list, indices, mode="max")
    image = np.where(image > 0.5, 255, 0)
    return image


def composite_patch_to_one_image_for_PossibilityMap(patch_image_folder, indices):
    filename = glob(patch_image_folder + "/*.npy")
    filename = SortNum(filename).sort_num()
    patch_list = []
    for i in filename:
        pre = np.load(i)
        patch_list.append(pre)
    image = em.merge_patches(patch_list, indices, mode="max")
    return image


def project_image(source_image_path, target_image_path, target_save_path, targetH, targetW, channel, scale=4,
                  lucc=None):
    """

    Args:
        source_image_path: image providing geo reference
        target_image_path: image needing geo reference or target_image
        target_save_path: the save path
        targetH: height of the target image
        targetW: width of the target image
        channel: channel number of the target image
        scale: size scale compared with the source image,default4. e.g., source is 64*64, target is 256*256
        lucc: the mask, which is a numpy array


    """
    source_image = rasterio.open(source_image_path)
    source_transform = source_image.get_transform()
    target_transform = source_transform.copy()
    target_transform[1] = target_transform[1] / scale
    target_transform[-1] = target_transform[-1] / scale
    target_transform = Affine.from_gdal(*target_transform)
    if type(target_image_path) == str:
        target_image = skimage.io.imread(target_image_path)  # the shape must be [C,H,W]
    else:
        target_image = target_image_path
    if isinstance(lucc, np.ndarray):
        target_image = target_image * lucc
    # target_image = target_image.astype("float32")
    # target_image[target_image == 0] = np.nan
    if len(target_image.shape) == 2:
        target_image = np.expand_dims(target_image, 0)  # [C,H,W]

    profile = {
        'driver': 'GTiff',
        'width': targetW,
        'height': targetH,
        'count': channel,
        'crs': 'EPSG:4326',  # '+proj=latlong'
        'transform': target_transform,
        'dtype': rasterio.uint8,
        'compress': 'LZW'  # to compress
    }

    with rasterio.open(target_save_path, 'w', **profile) as f:
        f.write(target_image)
    # print("The image is projected successfully!")
