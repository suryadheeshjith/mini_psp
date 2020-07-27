import os.path as osp
from io import StringIO
import numpy as np

from mini_psp.utils.metric_utils import evaluate, eval_conf_matrix
from mini_psp.utils.logger_utils import get_logger


def save_details(args,Inputs,Output):

    '''Saves details of patch generation run'''

    input_shape = np.array(Inputs).shape
    if(Output):
        target_shape = np.array(Output).shape

    save_path = osp.join(args.output_fol,"data_details.txt")
    f= open(save_path,"w+")

    f.write("\nData details : \n\n")
    f.write("Input folder : {}\n".format(osp.abspath(args.input_fol)))
    f.write("Output folder : {}\n".format(osp.abspath(args.output_fol)))
    f.write("Strides taken : {}\n".format(args.strides))

    if(Output and args.thresh>0 and args.percentage_ones>0):
        f.write("Selecting Patches with Percentage ones and threshold : {},{}\n".format(args.percentage_ones,args.thresh))

    elif((not Output) and (args.percentage_ones>0 and args.thresh>0)):
        f.write("There are no target files, hence no selection done. Threshold values ignored.\n")

    elif(Output and (args.thresh==0 or args.percentage_ones==0)):
        f.write("All Patches considered.\n")



    f.write('Input shape : {}\n'.format(input_shape))

    if(Output):
        f.write('Target shape : {}'.format(target_shape))
    f.close()


def get_summary_string(model):

    '''Returns the string of model.summary() in a temporary variable'''

    tmp_smry = StringIO()
    model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
    summary = tmp_smry.getvalue()
    return summary


def save_model(model, model_path):

    '''Saves model JSON and the model weights'''

    # Logger
    logger = get_logger()

    # Save model JSON
    logger.info("Saving model json")
    model_json = model.to_json()
    save_json_path = osp.join(model_path,"model.json")
    with open(save_json_path, "w") as json_file:
        json_file.write(model_json)

    # Save final weights
    logger.info("Saving model weights")
    save_weight_path = osp.join(model_path,"model_final_weights.h5")
    model.save_weights(save_weight_path)


def log_eval(y_test,y_pred,n_classes,cm):

    '''Logs evaluation metrics of all classes.'''

    # Logger
    logger = get_logger()

    # acc,iou,f1 = evaluate(y_test,y_pred,n_classes)
    # logger.info("Class\t\tAccuracy")
    # for k in acc.keys():
    #     logger.info("{}\t\t{}".format(k,acc[k]))

    iou,f1 = evaluate(y_test,y_pred,n_classes)
    logger.info("\n\n")
    logger.info("Class\t\tIoU")
    for k in iou.keys():
        logger.info("{}\t\t{}".format(k,iou[k]))

    logger.info("\n\n")
    logger.info("Class\t\tF1-Score")
    for k in f1.keys():
        logger.info("{}\t\t{}".format(k,f1[k]))

    logger.info("\nConfusion matrix : \n")
    logger.info(cm)
    logger.info("\n\n")
    ovAc, kappa, prod_acc, user_acc = eval_conf_matrix(cm,n_classes)

    logger.info("Class\t\tProducer Accuracy")
    for i,k in enumerate(prod_acc):
        logger.info("{}\t\t{}".format(i,k))
    logger.info("\n\n")
    logger.info("Class\t\tUser Accuracy")
    for i,k in enumerate(user_acc):
        logger.info("{}\t\t{}".format(i,k))

    logger.info("\nOverall Accuracy : {}".format(ovAc))
    logger.info("\nKappa Coefficient : {}".format(kappa))
