"""

This file is used to train the model on the data given as input and saves the JSON and weights files in the directory provided by 'Model path'. There is also
provision to set the number of epochs and batch size in the command line.

-------------------------------------------------------------------------------------------------------------
INPUT (Command line Arguments):
    * Input npy file path corresponding to the patches generated from the satellite images
    * Output npy file path corresponding to the patches generated from the target masks
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
-------------------------------------------------------------------------------------------------------------

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import os.path as osp

from mini_psp.utils.logger_utils import get_logger
from mini_psp.utils.plot_utils import plot_history
from mini_psp.utils.store_utils import get_summary_string, save_model
from mini_psp.models.models import psp_net, unet, fcn


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--inp",dest="input_npy", help="Input npy file path.",required = True)
    parser.add_argument('-o',"--out",dest="output_npy", help="Output npy file path.",required = True)
    parser.add_argument('-mp',"--mpath",dest="mpath", help="Model path to save all required files for testing.", required = True)
    parser.add_argument('-mn',"--mname",dest="mname", help="Model name. Options : psp, unet or fcn. Default = psp",default = 'psp')
    parser.add_argument('-e',"--epochs",dest="epochs", default=50, type = int, help="Number of epochs. Default = 50")
    parser.add_argument('-b',"--batch",dest="batch_size", default=8, type = int, help="Batch size. Default = 8")
    parser.add_argument('-tt',"--traintest",action="store_true", dest="train_test", default=False, help="Use Train Test split. Default = False")
    parser.add_argument('-pl',"--plot",action="store_true", dest="plot_hist", default=False, help="Plot Accuracy and Loss graphs. Default = False")
    args = parser.parse_args()
    return args


def get_model(model_name, input_shape, n_classes):

    '''Selects the model based on model_name'''

    # Logger
    logger = get_logger()

    # PSP
    if(model_name.lower()=='psp'):
        logger.info("PSPNet model used")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=100000,decay_rate=0.96,staircase=True)
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=False)
        loss = 'categorical_crossentropy'
        model = psp_net(input_shape=input_shape, optimizer = optimizer, loss = loss, n_classes=n_classes)

    # UNET
    elif(model_name.lower()=='unet'):
        logger.info("U-NET model used")
        optimizer = 'adam'
        loss = 'categorical_crossentropy'
        model = unet(input_shape=input_shape, optimizer = optimizer, loss = loss, n_classes=n_classes)

    # FCN
    elif(model_name.lower()=='fcn'):
        logger.info("FCN model used")
        optimizer = 'adam'
        loss = 'categorical_crossentropy'
        model = fcn(input_shape=input_shape, optimizer = optimizer, loss = loss, n_classes=n_classes)

    else:
        logger.info("Enter valid model name")
        exit(0)

    return model



def train(args):

    '''Train function'''


    model_list = ['unet','fcn','psp']

    # Logger
    logger = get_logger()


    # Checks
    assert args.mname in model_list
    if not osp.exists(args.mpath):
        os.makedirs(args.mpath)


    input_npy = args.input_npy
    output_npy = args.output_npy
    model_path = args.mpath
    model_name = args.mname

    # Load data
    x_dataset = np.load(input_npy)
    y_dataset = np.load(output_npy)
    dataset = [x_dataset, y_dataset]

    # Parameters
    assert x_dataset.shape[:-1] == y_dataset.shape[:-1]
    input_shape = x_dataset.shape[1:]
    n_classes = y_dataset.shape[-1]

    # Reshape for training
    dataset[1] = np.reshape(dataset[1],(-1,dataset[1].shape[1]*dataset[1].shape[2],n_classes))

    # Train test split
    if(args.train_test):
        logger.info("Splitting data into train test sets.")
        dataset[0], dataset[1] = shuffle(dataset[0],dataset[1],random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=42)
        dataset[0] = X_train
        dataset[1]= y_train

    # Save weights checkpoint
    save_best_weights_path = osp.join(model_path,"best_weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(save_best_weights_path, monitor='val_acc',verbose = 1,save_best_only=True, mode='max')
    early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    callbacks_list = [checkpoint]

    model = get_model(model_name, input_shape, n_classes)

    # Logging model.summary()
    summary  = get_summary_string(model)
    logger.info("Model Summary :\n {}".format(summary))

    # Model fit
    history = model.fit(dataset[0], dataset[1], epochs=args.epochs, batch_size =args.batch_size,validation_split=0.2,callbacks=callbacks_list)

    # Save model
    save_model(model, model_path)

    # Plot history option
    if(args.plot_hist):
        plot_history(history,args.mpath)



if __name__ == '__main__':

    # Parse Args
    args = parse_args()

    train(args)
