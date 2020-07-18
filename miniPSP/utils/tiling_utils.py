import rasterio
import rasterio.windows
from rasterio.windows import Window
import random
import numpy as np
import os
import os.path as osp

def create_patches(src: rasterio.DatasetReader, size_h, size_w, overlap=0):

    """
    Patches are created here.
    """

    if(overlap==0):
        num_patches_w = src.width // size_w
        num_patches_h = src.height // size_h

        patches = []
        for i in range(0, num_patches_w):
            for j in range(0, num_patches_h):
                patches.append({
                    "data": src.read(1, window=Window(i*size_w, j*size_h, size_w, size_h)),
                    "crs": src.crs,
                    "transform": rasterio.windows.transform(Window(i*size_w, j*size_h, size_w, size_h), src.transform)
                })

        return patches

    else:
        num_patches_w = (src.width - size_w) // overlap
        num_patches_h = (src.height - size_h) // overlap

        patches = []
        for i in range(0, num_patches_w):
            for j in range(0, num_patches_h):
                patches.append({
                    "data": src.read(1, window=Window(i*overlap, j*overlap, size_w, size_h)),
                    "crs": src.crs,
                    "transform": rasterio.windows.transform(Window(i*overlap, j*overlap, size_w, size_h), src.transform)
                })

        return patches


def select_patches(patchesX, patchesY, percentage_ones,random_thresh):

    """
    Patches are selected here based on a threshold given and the percentage of true pixels (value 1 pixels) required in each patch.
    """

    retX = []
    retY = []

    #random.seed(1)

    for i in range(len(patchesY)):
        count = np.sum(patchesY[i])
        if (count >= percentage_ones*((patchesY[i].shape[0]**2)*patchesY[i].shape[-1])) or random.randrange(0, 10) >= random_thresh:
            retX.append(patchesX[i])
            retY.append(patchesY[i])

    return retX, retY


def write_single_patch(data, out_file: str, height, width, crs, transform, windowI, windowJ):

    """
    Writes a single patch.
    """

    patch_width = data.shape[1]
    patch_height = data.shape[0]

    if windowI == windowJ == 0:
        st = 'w'
    else:
        st = 'r+'

    with rasterio.open(
        out_file,
        st,
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data, window=Window(windowJ*patch_width,
                                      windowI*patch_height, patch_width, patch_height), indexes=1)



def save_masks(save_path, y_pred):

    """
    Saving output prediction masks.
    """

    # A sample file used to provide various information to the predicted mask during save.
    sample_path = None
    sam_path = 'Data/Targets'
    for i in os.listdir(sam_path):
        if(i.endswith("tif")):
            sample_path = osp.join(sam_path,i)
            break
    if(not sample_path):
        print("No valid reference path!")
        exit(0)

    with rasterio.open(sample_path) as src:
        input_crs = src.crs
        input_transform = src.transform
        input_width = src.width
        input_height = src.height

    patches_per_row = input_height // y_pred.shape[1]

    for i in range(5):
        out = osp.join(save_path,'predicted_band{}.tif'.format(i+1))
        for j in range(0, len(y_pred)):
            write_single_patch(y_pred[j,:,:,i], out, input_height, input_width, input_crs, input_transform,j % patches_per_row, j // patches_per_row)
