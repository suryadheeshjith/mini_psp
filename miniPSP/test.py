"""

This file is used to test the model on the data given as input based on the JSON and weights files saved during training. The output is
based on command line arguments given by the user. For evaluation, Accuracy, IoU and F1-score is logged for each class with their means. The
confusion matrix and the output masks can also be saved.

-------------------------------------------------------------------------------------------------------------
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
    * Saved normalised confusion matrix
    * Saved output masks
-------------------------------------------------------------------------------------------------------------

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.keras.optimizers import *
import numpy as np
import argparse
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils.data_utils import round_outputs
from utils.model_utils import get_json
from utils.logger_utils import get_logger
from utils.metric_utils import conf_matrix
from utils.plot_utils import plot_confusion_matrix
from utils.store_utils import log_eval
from utils.tiling_utils import save_masks


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--inp",dest="input_npy", help="Input npy file path.",required = True)
    parser.add_argument('-o',"--out",dest="output_npy", help="Output npy file path.",required = True)
    parser.add_argument('-mj',"--mjpath",dest="mjpath", help="Model JSON file path.", required = True)
    parser.add_argument('-mw',"--mwpath",dest="mwpath", help="Model weights file path.", required = True)
    parser.add_argument('-mn',"--mname",dest="mname", help="Model name. Options : psp, unet or fcn. Default = psp",default = 'psp')
    parser.add_argument('-tt',"--traintest",action="store_true", dest="train_test", default=False, help="Use Train Test split. Default = False")
    parser.add_argument('-e',"--eval",action="store_true", dest="eval", default=False, help="Evaluate the model and log the results. Default = False")
    parser.add_argument('-pl',"--plot",action="store_true", dest="plot_conf", default=False, help="Plot confusion matrix. Default = False")
    parser.add_argument('-s',"--save",action="store_true", dest="save_masks", default=False, help="Save masks for each class. Default = False")
    args = parser.parse_args()
    return args



def test(args, class_names):

    '''Test function'''

    input_npy = args.input_npy
    output_npy = args.output_npy
    save_json_path = args.mjpath
    save_weight_path = args.mwpath
    model_path = osp.abspath(osp.dirname(args.mjpath))

    # Read model from JSON
    model = get_json(save_json_path)

    # Load weights
    model.load_weights(save_weight_path)

    # Model initialisation
    if(args.mname.lower()=='psp'):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=100000,decay_rate=0.96,staircase=True)
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = Adam()

    model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

    # Load data files
    X_test = np.load(input_npy)
    y_test = np.load(output_npy)

    X_test, y_test = shuffle(X_test,y_test,random_state=42)

    if(args.train_test):
        X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    # Predict
    y_pred = model.predict(X_test)


    # Reshape
    y_pred = np.reshape(y_pred,(-1,256,256,5))
    y_test = np.reshape(y_test,(-1,256,256,5))

    # Obtain most likely class for each pixel and set to value 1
    y_pred = round_outputs(y_pred)

    # Evaluate model
    if(args.eval):
        logger.info("Logging evaluated metrics")
        log_eval(y_test,y_pred,n_classes=len(class_names))

    # Confusion matrix
    if(args.plot_conf):
        cm = conf_matrix(y_test,y_pred)
        logger.info("\nConfusion matrix : ")
        logger.info(cm)
        plot_confusion_matrix(cm,class_names,model_path)

    # Save masks
    if(args.save_masks):
        logger.info("Saving masks")
        save_masks(model_path, y_pred)

if __name__ == '__main__':

    # Class names (Can be changed by User)
    class_names = ['Veg','Built-up','Open land','Roads','Waterbodies']


    # Logger
    logger = get_logger()

    # Parse Args
    args = parse_args()


    if not osp.exists(args.mjpath):
        print("Enter valid model json path")

    if not osp.exists(args.mwpath):
        print("Enter valid model weights path")

    test(args, class_names)
