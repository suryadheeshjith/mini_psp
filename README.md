
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

<img src="figures/WaterbodiesImage.PNG" alt="Waterbodies Image" width =200>
<img src="figures/Waterbodies.PNG" alt="Waterbodies Mask" width =200>


Apart from the PSPNet model, UNET and FCN models have also been implemented.


## Quick Start

1. git clone the repository.

    ```git clone https://github.com/suryadheeshjith/ISRO_Repo.git```

2. Create a new environment for this project [OPTIONAL]. If you use conda, create an environment using this command.

    ```conda create -n ISRO python=3.6.10```

3. Install the required packages from requirements.txt.

    ```pip install -r requirements.txt```


Required Packages :

* Python==3.6.10
* numpy==1.18.1
* argparse
* scikit-learn==0.22.1
* scipy==1.4.1
* rasterio==1.1.5
* matplotlib==3.2.1
* json5==0.8.5
* tensorflow==1.15.2
* logger==1.4


The repository structure contains the Data folder where all the images are to be kept.


    ```
    .
    ├── README.md
    ├── figures
    ├── miniPSP
    │   ├── Data
    │   │   ├── Bands
    │   │   ├── Targets
    │   ├── models
    │   │   ├── __init__.py
    │   │   └── models.py
    │   |── utils
    │   │   ├── __init__.py
    │   │   ├── data_utils.py
    │   │   ├── logger_utils.py
    │   │   ├── metric_utils.py
    │   │   ├── model_utils.py
    │   │   ├── plot_utils.py
    │   │   ├── store_utils.py
    │   │   └── tiling_utils.py
    │   ├── patch_generator.py
    │   ├── test.py
    │   ├── train.py
    └── requirements.txt
    ```

We first generate patches for the input and mask tif files and then train our model.


### Running demo code

To generate patches, we run the [patch_generator.py]() file

```python3 Main.py -d Kad -p DefaultParameters.json```

### List of Command-line Parameters

* -h --help : List the parameters and their use.

* -d --dataset : A dataset must be considered for learning. This parameter takes the dataset csv file name. This parameter **must** be passed.    

* -p --parameters : Model Parameters are passed using a json file. This parameter must be used to specify the name of json file. This parameter **must** be passed.  

* -i --ignore : Ignore the first column. (For some cases).  
                Default = False

* -r --randomsamp : Balance the dataset using random under sampling. (Use for imbalanced datasets).   
                    Default = False

* -v --parentvaluecols [ BETA ]: Addition of columns based on class distributions of parents of leaf nodes in the decision tree.    
                                Default = False

* -c --cores [ BETA ]: Number of cores to be used during addition of columns (When -v is True).    
                         Default = -1 (All cores)

To train the model



##### Contributors

1. Surya Dheeshjith
2. A. Suryanarayanan
3. Shyam A.

###### Last updated : 17 July 2020
