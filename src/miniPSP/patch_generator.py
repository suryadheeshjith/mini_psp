"""
This file is used to generate patches from the Satellite images and Target Masks. An input directory that contains all these files is accepted
as input and each file is parsed and patches are generated for training the model. All the files must be in .tif format. The input directory
must contain two folders Bands and Targets, and each must contain the Satellite image bands and Target Masks. For example, you could call your directory
'Data' and it must have a directory structure like this -

├── miniPSP
│   ├── Data
│   │   ├── Bands
│___│___├── Targets

The npy files will then be saved in the Output directory.
-------------------------------------------------------------------------------------------------------------
INPUT (Command line Arguments):
    * Directory containing the Satellite images and Target Masks (Optional). (.tif files)
    * Output Directory
    * Dimensions of patch size [OPTIONAL][DEFAULT=256]
    * Stride length [OPTIONAL][DEFAULT=0]
    * Threshold for selecting patches [OPTIONAL][DEFAULT=8]
    * Percentage ones for selecting patches [OPTIONAL][DEFAULT=0]
    * Percentage ones for selecting patches [OPTIONAL][DEFAULT=0]
    * Option for separate train test files [OPTIONAL][DEFAULT=False]
    * Option for saving details of saved .npy files [OPTIONAL][DEFAULT=False]

OUTPUT :
    Npy files corresponding to the input. An optional data_details text file corresponding to the details of saved files.
-------------------------------------------------------------------------------------------------------------


NOTE : If targets need not be patched, you need not include any .tif files in the Targets folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

from miniPSP.utils.data_utils import save_npy
from miniPSP.utils.logger_utils import get_logger


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',"--tdim",dest="tdim", default=256,type = int, help="Dimension of the patch size (Height/Width). Default = 256")
    parser.add_argument('-i',"--inpf",dest="input_fol", help="Input Folder containing the input tiff files.",required = True)
    parser.add_argument('-o',"--outf",dest="output_fol", help="Output folder to store the training .npy files.",required = True)
    parser.add_argument('-tr',"--threshr", dest="thresh", type = int, default=8, help="Random threshold for selecting patches. Enter value between 0 - 10 Default = 8")
    parser.add_argument('-tp',"--threshp", dest="percentage_ones", type = float, default=0.25, help="Lower bound of true values in target masks for selecting patches. Enter float value between 0 - 1 Default = 0.0")
    parser.add_argument('-str',"--strides", dest="strides", type = int, default=0, help="Strides taken for tiling to obtain overlapping patches. Default = 0 (for non-overlapping patches)")
    parser.add_argument('-tt',"--traintest",action="store_true", dest="train_test", default=False, help="Save separate files for training and testing. Default = False")
    parser.add_argument('-s',"--save",action="store_true", dest="save_details", default=False, help="Save details of patches generated. Default = False")
    args = parser.parse_args()
    return args


def generate(args):

    '''Generates Patches'''

    # Logger
    logger = get_logger()

    logger.info("Command Details : ")
    logger.info(json.dumps(vars(args), indent=4))

    #Save files
    save_npy(args)


if __name__ == '__main__':

    # Parse Args
    args = parse_args()

    generate(args)
