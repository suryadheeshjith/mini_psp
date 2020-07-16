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
from io import StringIO

from utils.logger_utils import get_logger
from utils.plot_utils import plot_history
from models.models import PSP_Net, UNET, FCN



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



def train(args):

    input_npy = args.input_npy
    output_npy = args.output_npy
    model_path = args.mpath
    model_name = args.mname

    # Load data
    x_dataset = np.load(input_npy)
    y_dataset = np.load(output_npy)
    dataset = [x_dataset, y_dataset]
    dataset[1] = np.reshape(dataset[1],(-1,dataset[1].shape[1]*dataset[1].shape[2],5))

    # Shuffle
    dataset[0], dataset[1] = shuffle(dataset[0],dataset[1],random_state=42)

    # Train test split
    if(args.train_test):
        X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=42)
        dataset[0] = X_train
        dataset[1]= y_train

    # Save weights checkpoint
    save_best_weights_path = osp.join(model_path,"best_weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(save_best_weights_path, monitor='val_acc',verbose = 1,save_best_only=True, mode='max')
    early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    callbacks_list = [checkpoint]

    # PSP
    if(model_name.lower()=='psp'):
        print("PSPNet model used")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=100000,decay_rate=0.96,staircase=True)
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=False)
        loss = 'categorical_crossentropy'
        model = PSP_Net(input_shape=(256,256,4), optimizer = optimizer, loss = loss, n_classes=5)

    # UNET
    elif(model_name.lower()=='unet'):
        print("UNET model used")
        optimizer = 'adam'
        loss = 'categorical_crossentropy'
        model = UNET4(input_shape=(256,256,4), optimizer = optimizer, loss = loss, n_classes=5)

    # FCN
    elif(model_name.lower()=='fcn'):
        print("FCN model used")
        optimizer = 'adam'
        loss = 'categorical_crossentropy'
        model = fcn_8(input_shape=(256,256,4), optimizer = optimizer, loss = loss, n_classes=5)

    else:
        print("Enter valid model name")
        exit(0)

    # Logging model.summary()
    tmp_smry = StringIO()
    model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
    summary = tmp_smry.getvalue()
    logger.info("Model Summary :\n {}".format(summary))
    history = model.fit(dataset[0], dataset[1], epochs=args.epochs, batch_size =args.batch_size,validation_split=0.2,callbacks=callbacks_list)

    # Save model JSON
    print("saving model json")
    model_json = model.to_json()
    save_json_path = osp.join(model_path,"model.json")
    with open(save_json_path, "w") as json_file:
        json_file.write(model_json)

    # Save final weights
    print("saving weights.h5")
    save_weight_path = osp.join(model_path,"model_final_weights.h5")
    model.save_weights(save_weight_path)

    # Plot history option
    if(args.plot_hist):
        plot_history(history,args.mpath)



if __name__ == '__main__':

    model_list = ['unet','fcn','psp']

    # Logger
    logger = get_logger()

    # Parse Args
    args = parse_args()


    assert args.mname in model_list

    if not osp.exists(args.mpath):
        os.makedirs(args.mpath)

    train(args)
