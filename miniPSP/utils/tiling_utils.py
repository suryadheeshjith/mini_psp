import rasterio
import rasterio.windows
from rasterio.windows import Window
import random
import numpy as np
import os.path as osp

def createTiles(src: rasterio.DatasetReader, size_h, size_w, overlap=0):

    if(overlap==0):
        num_tiles_w = src.width // size_w
        num_tiles_h = src.height // size_h

        tiles = []
        for i in range(0, num_tiles_w):
            for j in range(0, num_tiles_h):
                tiles.append({
                    "data": src.read(1, window=Window(i*size_w, j*size_h, size_w, size_h)),
                    "crs": src.crs,
                    "transform": rasterio.windows.transform(Window(i*size_w, j*size_h, size_w, size_h), src.transform)
                })

        return tiles

    else:
        num_tiles_w = (src.width - size_w) // overlap
        num_tiles_h = (src.height - size_h) // overlap

        tiles = []
        for i in range(0, num_tiles_w):
            for j in range(0, num_tiles_h):
                tiles.append({
                    "data": src.read(1, window=Window(i*overlap, j*overlap, size_w, size_h)),
                    "crs": src.crs,
                    "transform": rasterio.windows.transform(Window(i*overlap, j*overlap, size_w, size_h), src.transform)
                })

        return tiles


def selectTiles(tilesX, tilesY, percentage_ones,random_thresh):

    retX = []
    retY = []

    #random.seed(1)

    for i in range(len(tilesY)):
        count = np.sum(tilesY[i])
        if (count >= percentage_ones*((tilesY[i].shape[0]**2)*tilesY[i].shape[-1])) or random.randrange(0, 10) >= random_thresh:
            retX.append(tilesX[i])
            retY.append(tilesY[i])

    return retX, retY


def writeTIFF(data, out_file: str, height, width, crs, transform, windowI, windowJ):

    tileWidth = data.shape[1]
    tileHeight = data.shape[0]

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
        dst.write(data, window=Window(windowJ*tileWidth,
                                      windowI*tileHeight, tileWidth, tileHeight), indexes=1)



def save_masks(sample_path,save_path, y_pred):
    with rasterio.open(sample_path) as src:
        input_crs = src.crs
        input_transform = src.transform
        input_width = src.width
        input_height = src.height

    tiles_per_row = input_height // y_pred.shape[1]

    for i in range(5):
        out = osp.join(save_path,'predicted_band{}.tif'.format(i+1))
        for j in range(0, len(y_pred)):
            writeTIFF(y_pred[j,:,:,i], out, input_height, input_width, input_crs, input_transform,j % tiles_per_row, j // tiles_per_row)
