from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

from utils.data_utils import save_npy


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',"--tdim",dest="tdim", default=256,type = int, help="Dimensions of the Tile size. Default = 256")
    parser.add_argument('-i',"--inpf",dest="input_fol", help="Input Folder containing the input tiff files.",required = True)
    parser.add_argument('-o',"--outf",dest="output_fol", help="Output folder to store the training .npy files.",required = True)
    parser.add_argument('-tp',"--threshp", dest="percentage_ones", type = float, default=0.25, help="Percentage ones in each tile. Enter value between 0 - 1 Default = 0.25")
    parser.add_argument('-tr',"--threshr", dest="thresh", type = int, default=8, help="Threshold parameter while selecting tiles. Enter value between 0 - 10 Default = 8")
    parser.add_argument('-str',"--strides", dest="strides", type = int, default=0, help="Strides taken for tiling to obtain overlapping patches. Default = 0 (for non-overlapping patches)")
    parser.add_argument('-s',"--save",action="store_true", dest="save_details", default=False, help="Save details of patches generated. Default = False")
    args = parser.parse_args()
    return args


def main():

    #Parse Args
    args = parse_args()

    print("Command Details : ")
    print(json.dumps(vars(args), indent=4))
    
    #Save files
    save_npy(args)


if __name__ == '__main__':
    main()
