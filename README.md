# ISRO_Repo

# mini-PSPNet for Semantic segmentation of LULC (TITLE)

# Introduction

This is the official implementation of [mini-PSPNet for Semantic segmentation of LULC](). We perform semantic segmentation on the city of Bengaluru, Karnataka, India. Implementation is done in Python 3.6.10.


## Details

The architecture of the model is shown below.

![Architecture](/figures/Architecture.png)




### Requirements

* All required packages are in requirements.txt

```pip install -r requirements.txt```



### Running demo code

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

### How to run code for different datasets and model parameters

```python3 Main.py -d <Dataset_Name> -p <Parameter_list>.json```




##### Contributors

1. Surya Dheeshjith
2. A. Suryanarayanan
3. Shyam A.

###### Last updated : 17 July 2020
