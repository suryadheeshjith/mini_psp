import numpy as np
import rasterio
import os
import os.path as osp

from .store_utils import save_details
from .tiling_utils import createTiles, selectTiles




def normalise_inputs(Inputs):
    Inputs = np.array(Inputs)
    nInputs = np.zeros_like(Inputs)
    np.divide(Inputs,np.max(Inputs,axis=(0,1,2),keepdims=True),out=nInputs,where=np.max(Inputs,axis=(0,1,2),keepdims=True)!=0)
    return nInputs


def get_multi_io(ls, w, h,overlap=False):

    Tiles = []
    inputs = []
    input1 = np.zeros(w*h*len(ls))
    input1 = input1.reshape(h, w, len(ls))

    for file_path in ls: # For each band's file path
        with rasterio.open(file_path) as src:
            _tiles = createTiles(src, h, w,overlap)
            Tiles.append(_tiles)

    for i in range(len(Tiles[0])):
        for j in range(len(Tiles)):
            input1[:, :, j] = Tiles[j][i]['data']

        inputs.append(input1)
        input1 = np.zeros((w,h,len(ls))) # Hack

    return inputs



def save_npy(args):


    # Getting input file names
    band_files = []
    target_files = []
    band_dir = args.input_fol +"/"+ "Bands"
    target_dir = args.input_fol +"/"+ "Targets"

    # Storing file paths
    for band_file in os.listdir(band_dir):
        if(band_file.endswith(".tif")):
            band_files.append(band_dir+"/"+band_file)
        else:
            print("File not considered : "+band_file + " in "+band_dir)

    for mask_file in os.listdir(target_dir):
        if(mask_file.endswith(".tif")):
            target_files.append(target_dir+"/"+mask_file)
        else:
            print("File not considered : "+mask_file + " in "+target_dir)


    if(not band_files):
        print("No relevant files in Band directory")
        exit(0)

    if(not target_files):
        print("No relevant files in Target directory")
        exit(0)


    width = args.tdim
    height = args.tdim

    # Overlapping or non-overlapping patches
    if(args.strides>0):
        Inputs = get_multi_io(band_files,width,height,overlap=args.strides)
        Output = get_multi_io(target_files,width,height,overlap=args.strides)

    else:
        Inputs = get_multi_io(band_files,width,height,overlap=0)
        Output = get_multi_io(target_files,width,height,overlap=0)

    # Normalising Inputs
    Inputs = normalise_inputs(Inputs)


    # Thresholding
    if(args.thresh>0):
        Inputs,Output = selectTiles(Inputs,Output,args.percentage_ones,args.thresh)


    #Saving input
    input_path = args.output_fol+"/"+'input'
    target_path = args.output_fol+"/"+'output'
    np.save(input_path,Inputs)
    np.save(target_path,Output)


    if(args.save_details):
        print("Saving data details to data_details.txt in "+args.output_fol)
        save_details(args,np.array(Inputs).shape,np.array(Output).shape)
