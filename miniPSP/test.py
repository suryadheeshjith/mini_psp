from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import load_model,model_from_json
import numpy as np
import argparse
import os
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils.data_utils import round_outputs
from utils.logger_utils import get_logger
from utils.metric_utils import evaluate, conf_matrix
from utils.plot_utils import plot_confusion_matrix
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

    input_npy = args.input_npy
    output_npy = args.output_npy
    save_json_path = args.mjpath
    save_weight_path = args.mwpath
    model_path = osp.abspath(osp.dirname(args.mjpath))

    # Read model from JSON
    json_file = open(save_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights
    loaded_model.load_weights(save_weight_path)
    if(args.mname.lower()=='psp'):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=100000,decay_rate=0.96,staircase=True)
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=False)
    else:
        optimizer = Adam()
    loaded_model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
    model = loaded_model

    # Load data files
    X_test = np.load(input_npy)
    y_test = np.load(output_npy)

    X_test, y_test = shuffle(X_test,y_test,random_state=42)

    if(args.train_test):
        X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.reshape(y_pred,(-1,256,256,5))
    y_test = np.reshape(y_test,(-1,256,256,5))

    # Obtain most likely class for each pixel and set to value 1
    y_pred = round_outputs(y_pred)

    # Evaluate model
    if(args.eval):
        acc,iou,f1 = evaluate(y_test,y_pred,n_classes=len(class_names))
        logger.info("Class\t\tAccuracy")
        for k in acc.keys():
            logger.info("{}\t\t{}".format(k,acc[k]))

        logger.info("\n\n")
        logger.info("Class\t\tIoU")
        for k in iou.keys():
            logger.info("{}\t\t{}".format(k,iou[k]))

        logger.info("\n\n")
        logger.info("Class\t\tF1-Score")
        for k in f1.keys():
            logger.info("{}\t\t{}".format(k,f1[k]))


    if(args.plot_conf):
        cm = conf_matrix(y_test,y_pred)
        logger.info("\nConfusion matrix : ")
        logger.info(cm)
        plot_confusion_matrix(cm,class_names,model_path)


    if(args.save_masks):
        sample_path = None
        sam_path = 'Data/Targets'
        for i in os.listdir(sam_path):
            if(i.endswith("tif")):
                sample_path = osp.join(sam_path,i)
                break
        if(not sample_path):
            print("No valid reference path!")
            exit(0)
        save_masks(sample_path, model_path, y_pred)

if __name__ == '__main__':


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
