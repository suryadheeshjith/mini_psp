import numpy as np
import rasterio
import os
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .store_utils import save_details
from .tiling_utils import create_patches, select_patches
from utils.logger_utils import get_logger


def normalise_inputs(Inputs):

    '''Normalising inputs across bands.'''

    Inputs = np.array(Inputs)
    nInputs = np.zeros_like(Inputs)
    np.divide(Inputs,np.max(Inputs,axis=(0,1,2),keepdims=True),out=nInputs,where=np.max(Inputs,axis=(0,1,2),keepdims=True)!=0)
    return nInputs


def get_multi_io(ls, w, h,overlap=False):

    '''Patches are given appropriate shape here.'''

    Patches = []
    inputs = []
    input1 = np.zeros(w*h*len(ls))
    input1 = input1.reshape(h, w, len(ls))

    for file_path in ls: # For each band's file path
        with rasterio.open(file_path) as src:
            _patches = create_patches(src, h, w,overlap)
            Patches.append(_patches)

    for i in range(len(Patches[0])):
        for j in range(len(Patches)):
            input1[:, :, j] = Patches[j][i]['data']

        inputs.append(input1)
        input1 = np.zeros((w,h,len(ls)))

    return inputs


def get_input_file_names(inp_fol):

    '''Input file names are obtained here.'''

    # Logger
    logger = get_logger()

    # Getting input file names
    band_files = []
    target_files = []
    band_dir = osp.join(inp_fol,"Bands")
    target_dir = osp.join(inp_fol,"Targets")

    # Storing file paths
    for band_file in os.listdir(band_dir):
        if(band_file.endswith(".tif")):
            band_files.append(osp.join(band_dir,band_file))
        else:
            logger.info("File not considered : "+band_file + " in "+band_dir)
    band_files.sort()
    logger.info("Band Files : {}".format(band_files))
    for mask_file in os.listdir(target_dir):
        if(mask_file.endswith(".tif")):
            target_files.append(osp.join(target_dir,mask_file))
        else:
            logger.info("File not considered : "+mask_file + " in "+target_dir)

    if(target_files):
        target_files.sort()
        logger.info("Target Files : {}".format(target_files))


    if(not band_files):
        logger.info("No relevant files in Band directory")
        exit(0)

    if(not target_files):
        logger.info("No relevant targets given, only patching satellite images... ")


    return band_files, target_files

def save_npy(args):

    '''Npy files are saved here.'''

    # Logger
    logger = get_logger()

    # Input file names
    band_files, target_files = get_input_file_names(args.input_fol)

    width = args.tdim
    height = args.tdim

    # Overlapping or non-overlapping patches
    if(args.strides>0):
        Inputs = get_multi_io(band_files,width,height,overlap=args.strides)
        if(target_files):
            Output = get_multi_io(target_files,width,height,overlap=args.strides)

    else:
        Inputs = get_multi_io(band_files,width,height,overlap=0)
        if(target_files):
            Output = get_multi_io(target_files,width,height,overlap=0)

    # Normalising Inputs
    Inputs = normalise_inputs(Inputs)

    # Setting Output to False if empty Targets folder
    if(not target_files):
        Output = False

    # Selecting patches
    if(Output and args.thresh>0 and args.percentage_ones>0):
        Inputs,Output = select_patches(Inputs,Output,args.percentage_ones,args.thresh)

    elif((not Output) and (args.percentage_ones>0 and args.thresh>0)):
        logger.info("There are no target files, hence no selection done. Ignoring threshold values...")

    elif(Output and (args.thresh==0 or args.percentage_ones==0)):
        logger.info("All Patches considered.\n")

    #Saving input
    if(not osp.exists(args.output_fol)):
        os.makedirs(args.output_fol)

    # Saving separate files for training and testing if required. (Can help avoid memory crashes)
    if(args.train_test):
        if(Output):
            Inputs, Output = shuffle(Inputs, Output,random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(Inputs, Output, test_size=0.2, random_state=42)
            np.save(osp.join(args.output_fol,'input8_train.npy'),X_train)
            np.save(osp.join(args.output_fol,'output8_train.npy'),y_train)
            np.save(osp.join(args.output_fol,'input8_test.npy'),X_test)
            np.save(osp.join(args.output_fol,'output8_test.npy'),y_test)
        else:
            logger.info("No target files for train test save! Please add relevant files.")
            exit("0")


    else:
        np.save(osp.join(args.output_fol,'input'),Inputs)
        if(Output):
            np.save(osp.join(args.output_fol,'output'),Output)



    if(args.save_details):
        logger.info("Saving data details to data_details.txt in "+args.output_fol)
        save_details(args,Inputs,Output)



def round_outputs(y_pred, n_classes):

    '''Rounding is done across bands. The class with the most likelihood is given a value of 1 and the rest 0.'''

    y_pred = np.reshape(y_pred,(-1,n_classes))
    for i in range(y_pred.shape[0]):
        tem = y_pred[i]
        best=0
        idx=-1
        for j in range(n_classes):
            if(tem[j]>best):
                best = tem[j]
                idx=j

        y_pred[i] = [0]*n_classes
        y_pred[i][idx] = 1
    y_pred = np.reshape(y_pred,(-1,256,256,n_classes))
    return y_pred
