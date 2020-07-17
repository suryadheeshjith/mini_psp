from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

from utils.data_utils import save_npy
from utils.logger_utils import get_logger


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',"--tdim",dest="tdim", default=256,type = int, help="Dimension of the patch size (Height/Width). Default = 256")
    parser.add_argument('-i',"--inpf",dest="input_fol", help="Input Folder containing the input tiff files.",required = True)
    parser.add_argument('-o',"--outf",dest="output_fol", help="Output folder to store the training .npy files.",required = True)
    parser.add_argument('-tp',"--threshp", dest="percentage_ones", type = float, default=0.25, help="Percentage ones in each tile. Enter value between 0 - 1 Default = 0.25")
    parser.add_argument('-tr',"--threshr", dest="thresh", type = int, default=8, help="Threshold parameter while selecting tiles. Enter value between 0 - 10 Default = 8")
    parser.add_argument('-str',"--strides", dest="strides", type = int, default=0, help="Strides taken for tiling to obtain overlapping patches. Default = 0 (for non-overlapping patches)")
    parser.add_argument('-tt',"--traintest",action="store_true", dest="train_test", default=False, help="Save separate files for training and testing. Default = False")
    parser.add_argument('-s',"--save",action="store_true", dest="save_details", default=False, help="Save details of patches generated. Default = False")
    args = parser.parse_args()
    return args


def generate():

    """

    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    INPUT : * Directory containing the Satellite images and Target Masks. (.tif files)
            * Output Directory
            * Dimensions of patch size
            * Stride length

    OUTPUT : Two npy files called input.npy and output.npy (stored in the output directory) corresponding to the patches generated from the satellite images
             and the target masks.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------

    The input directory is taken and all the files are parsed and patches are generated for training the model. All the files must be in .tif format. This
    directory must contain two folders Bands and Targets, and each must contain their respective files. For example, you could call your directory Data and
    it must contain a structure like this -

    ├── miniPSP
    │   ├── Data
    │   │   ├── Bands
    │___│___├── Targets

    The npy files will then be saved in the Output directory.

    """

    # Logger
    logger = get_logger()

    # Parse Args
    args = parse_args()

    logger.info("Command Details : ")
    logger.info(json.dumps(vars(args), indent=4))

    #Save files
    save_npy(args)


if __name__ == '__main__':
    generate()
