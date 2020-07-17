import os.path as osp
from io import StringIO

from .metric_utils import evaluate
from .logger_utils import get_logger


def save_details(args,input_shape,target_shape):

    save_path = args.output_fol+"/"+"data_details.txt"
    f= open(save_path,"w+")

    f.write("\nData details : \n\n")
    f.write("Input folder : {}\n".format(osp.abspath(args.input_fol)))
    f.write("Output folder : {}\n".format(osp.abspath(args.output_fol)))
    f.write("Strides taken : {}\n".format(args.strides))

    if(args.thresh>0):
        f.write("Selecting Tiles with Percentage ones and threshold : {},{}\n".format(args.percentage_ones,args.thresh))


    f.write('Input shape : {}\n'.format(input_shape))
    f.write('Target shape : {}'.format(target_shape))
    f.close()


def get_summary_string(model):
    tmp_smry = StringIO()
    model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
    summary = tmp_smry.getvalue()
    return summary


def save_model(model, model_path):

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


def log_eval(y_test,y_pred,n_classes):

    # Logger
    logger = get_logger()

    acc,iou,f1 = evaluate(y_test,y_pred,n_classes)
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
