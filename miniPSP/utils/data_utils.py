import numpy as np
import rasterio
import os
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .store_utils import save_details
from .tiling_utils import createTiles, selectTiles
from utils.logger_utils import get_logger


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

    # Logger
    logger = get_logger()

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
            logger.info("File not considered : "+band_file + " in "+band_dir)

    for mask_file in os.listdir(target_dir):
        if(mask_file.endswith(".tif")):
            target_files.append(target_dir+"/"+mask_file)
        else:
            logger.info("File not considered : "+mask_file + " in "+target_dir)


    if(not band_files):
        logger.info("No relevant files in Band directory")
        exit(0)

    if(not target_files):
        logger.info("No relevant files in Target directory")
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
    if(not osp.exists(args.output_fol)):
        os.makedirs(args.output_fol)

    # Saving separate files for training and testing if required. (Can help avoid memory crashes)
    if(args.train_test):
        X_train, X_test, y_train, y_test = train_test_split(Inputs, Output, test_size=0.2, random_state=42)
        np.save(args.output_fol+"/"+'input8_train.npy',X_train)
        np.save(args.output_fol+"/"+'output8_train.npy',y_train)
        np.save(args.output_fol+"/"+'input8_test.npy',X_test)
        np.save(args.output_fol+"/"+'output8_test.npy',y_test)


    else:
        np.save(args.output_fol+"/"+'input',Inputs)
        np.save(args.output_fol+"/"+'output',Output)



    if(args.save_details):
        logger.info("Saving data details to data_details.txt in "+args.output_fol)
        save_details(args,np.array(Inputs).shape,np.array(Output).shape)



def round_outputs(y_pred):
    y_pred = np.reshape(y_pred,(-1,5))
    for i in range(y_pred.shape[0]):
        tem = y_pred[i]
        best=0
        idx=-1
        for j in range(5):
            if(tem[j]>best):
                best = tem[j]
                idx=j

        y_pred[i] = [0]*5
        y_pred[i][idx] = 1
    return y_pred
