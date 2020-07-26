
# Mini-PSPNet for Urban Land-Use/Land-Cover Classification of Remote Sensing images

## Introduction

This is the official implementation of [Mini-PSPNet for Urban Land-Use/Land-Cover Classification of Remote Sensing images](). We perform semantic segmentation on the city of Bengaluru, Karnataka, India and the implementation is done in Python 3.6.10.


## Details

This model uses a modified PSPNet that contains around 700,000 parameters and it's architecture is shown below.

<img src="figures/Architecture.png" alt="Architecture" width =600>

Each pixel is classified into one of the following classes :
1. Vegetation
2. Built-up
3. Open land
4. Roads
5. Waterbodies

The proposed study makes use of multispectral satellite imagery collected by the Sentinel-2 mission. The data collected was of 16-bit 6 type, covering the study area, and four 10 m spatial resolution bands of Sentinel-2, i.e., B2 (490 nm), B3 (560 nm), B4 (665 nm) and B8 (842 nm) are considered. The input's dimensions are 256 x 256 x 4, which means we use a height and width of 256. Some examples are given below.

&nbsp; &nbsp; <img src="figures/WaterbodiesImage.PNG" alt="Waterbodies Image" width =150> &nbsp;&nbsp;<img src="figures/Waterbodies.PNG" alt="Waterbodies Mask" width =150>&nbsp;&nbsp; <img src="figures/RoadsImage.PNG" alt="Roads Image" width =150> &nbsp;&nbsp;<img src="figures/Roads.PNG" alt="Roads Mask" width =150>

&nbsp; &nbsp; <img src="figures/OpenlandImage.PNG" alt="Openland Image" width =150>&nbsp;&nbsp; <img src="figures/Openland.PNG" alt="Openland Mask" width =150> &nbsp;&nbsp;<img src="figures/VegetationImage.PNG" alt="Vegetation Image" width =150> &nbsp;&nbsp;<img src="figures/Vegetation.PNG" alt="Vegetation Mask" width =150>

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="figures/BuildingImage.PNG" alt="Built-up Image" width =150> &nbsp;&nbsp;<img src="figures/Buildings.PNG" alt="Built-up Mask" width =150>


Apart from the PSPNet model, UNET and FCN models have also been implemented.


## Quick Start

### Install

1. git clone the repository.

    ```git clone https://github.com/suryadheeshjith/ISRO_Repo.git```

2. [OPTIONAL STEP] Create a new environment for this project. If you use conda, create an environment using this command.

    ```conda create -n ISRO python=3.6.10```

3. Enter the cloned directory

    ```cd ISRO_Repo```

4. Install the required packages

    ```python3 setup.py install```



#### Repository Structure

    .
    ├── README.md
    ├── figures
    ├── miniPSP
    │   ├── Data
    │   │   ├── Bands
    │   │   └── Targets
    │   ├── models
    │   │   ├── __init__.py
    │   │   └── models.py
    │   ├── patch_generator.py
    │   ├── test.py
    │   ├── train.py
    │   └── utils
    │       ├── __init__.py
    │       ├── data_utils.py
    │       ├── logger_utils.py
    │       ├── metric_utils.py
    │       ├── model_utils.py
    │       ├── plot_utils.py
    │       ├── store_utils.py
    │       └── tiling_utils.py
    ├── requirements.txt
    └── setup.py


The Data folder contains the Bands and the Targets folders. The Bands folder should contain the satellite image tiff files and the Targets should contain the target masks for each class.

### Running Model

Once all the files are in place, we crop our images into patches and then perform training and testing. So first off, change directories to the /miniPSP directory.

```cd miniPSP```



### Patch Generation

To generate patches, we run the [patch_generator.py](https://github.com/suryadheeshjith/ISRO_Repo/blob/master/miniPSP/patch_generator.py) file. This file is used to generate patches from the Satellite images and Target Masks. An input directory that contains all these files is accepted as input and each file is parsed and patches are generated for training the model. All the files must be in .tif format. The input directory must contain two folders Bands and Targets, and each must contain the Satellite image bands and Target Masks. For example, you could call your directory 'Data' and it must have a directory structure like this -

    ├── miniPSP
    │   ├── Data
    │   │   ├── Bands
    │___│___├── Targets

The npy files will then be saved in the Output directory.


INPUT (Command line Arguments):
* Directory containing the Satellite images and Target Masks (Optional). (.tif files)
* Output Directory
* Dimensions of patch size [OPTIONAL][DEFAULT=256]
* Stride length [OPTIONAL][DEFAULT=0]
* Random Threshold for selecting patches. Enter value between 0-10. [OPTIONAL][DEFAULT=8]
* Lower bound of true values in target masks for selecting patches. Enter float value between 0 - 1 [OPTIONAL][DEFAULT=0.0]
* Option for separate train test files [OPTIONAL][DEFAULT=False]
* Option for saving details of saved .npy files [OPTIONAL][DEFAULT=False]

OUTPUT :
* Npy files corresponding to the input. An optional data_details text file corresponding to the details of saved files.

NOTE : If targets need not be patched, you need not include any .tif files in the Targets folder.

An example command would be

```python3 patch_generator.py -i Data -o Data -s```


### Model Training

Training the model will save a JSON file, a best weights and final weights file. Training is done by the [train.py](https://github.com/suryadheeshjith/ISRO_Repo/blob/master/miniPSP/train.py) file. This file is used to train the model on the data given as input and saves the JSON and weights files in the directory provided by 'Model path'. There is also provision to set the number of epochs and batch size in the command line.


INPUT (Command line Arguments):
* Input npy file path corresponding to the patches generated from the satellite images
* Output npy file path corresponding to the patches generated from the target masks. [OPTIONAL]
* Model path
* Model name [OPTIONAL][DEFAULT='psp']
* Number of Epochs [OPTIONAL][DEFAULT=50]
* Batch Size [OPTIONAL][DEFAULT=8]
* Train Tested Data used [OPTIONAL][DEFAULT=False]
* Evaluate the model and log the results [OPTIONAL][DEFAULT=False]
* Save Accuracy and Loss graphs [OPTIONAL][DEFAULT=False]

OUTPUT :
* Model JSON file
* Model Weights file (Best weights and Final weights)


An example command would be

```python3 train.py -i Data/input.npy -o Data/output.npy -mp Model_test -tt -pl```

### Model Testing

Testing is done by the [test.py](https://github.com/suryadheeshjith/ISRO_Repo/blob/master/miniPSP/test.py) file. This file is used to test the model on the data given as input based on the JSON and weights files saved during training. The output is based on command line arguments given by the user. For evaluation, Accuracy, IoU and F1-score is logged for each class with their means. The confusion matrix and the output masks can also be saved.


INPUT (Command line Arguments):
* Input npy file path corresponding to the patches generated from the satellite images.
* Output npy file path corresponding to the patches generated from the target masks.
* Model JSON path
* Model weights path
* Model name [OPTIONAL][DEFAULT='psp']
* Train Tested Data used [OPTIONAL][DEFAULT=False]
* Evaluate the model and log the results [OPTIONAL][DEFAULT=False]
* Plot confusion matrix [OPTIONAL][DEFAULT=False]
* Save masks for each class [OPTIONAL][DEFAULT=False]

OUTPUT :
* Evaluate the model based on Accuracy, IoU and F1-score
* Saved confusion matrix
* Saved output masks


An example command would be

```python3 test.py -mj Model_test/model.json -i Data/input.npy -o Data/output.npy -mw Model_test/model_final_weights.h5 -tt -e -pl```

## Generating Masks

For prediction and subsequent generation of masks by the model, you must first generate patches for the entire image without any sampling or thresholding. This can be done by command

```python3 patch_generator.py -i Data -o Data -s -tp 0```

Then, to generate the masks

```python3 test.py -mj Model_test/model.json -i Data/input.npy -o Data/output.npy -mw Model_test/model_final_weights.h5 -s```


##### Contributors

1. Surya Dheeshjith
2. A. Suryanarayanan
3. Shyam A.

###### Last updated : 17 July 2020
