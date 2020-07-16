import numpy as np
from sklearn import metrics


def get_iou(target,prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def get_mean_accuracy(target,prediction):
    return np.mean(target==prediction)

def get_class_accuracies(target,prediction,n_classes):

    assert len(target.shape)==4
    assert len(prediction.shape)==4

    sum =0
    acc = {}
    for i in range(n_classes):
        cur_acc = np.mean(prediction[:,:,:,i]==target[:,:,:,i])
        sum+=cur_acc
        acc[i+1] = cur_acc
    acc['mean'] = sum/5
    return acc


def get_class_IoU(target,prediction,n_classes):

    assert len(target.shape)==4
    assert len(prediction.shape)==4

    sum =0
    IoU = {}
    for i in range(n_classes):
        cur_iou = get_iou(prediction[:,:,:,i],target[:,:,:,i])
        sum+=cur_iou
        IoU[i+1] = cur_iou
    IoU['mean'] = sum/5
    return IoU


def get_class_F1(target,prediction,n_classes):

    assert len(target.shape)==4
    assert len(prediction.shape)==4

    sum =0
    f1 = {}
    for i in range(n_classes):
        cur_f1 = metrics.f1_score(prediction[:,:,:,i].reshape(-1,1),target[:,:,:,i].reshape(-1,1))
        sum+=cur_f1
        f1[i+1] = cur_f1
    f1['mean'] = sum/5
    return f1


def evaluate(target,prediction,n_classes):
    acc = get_class_accuracies(target,prediction,n_classes)
    iou = get_class_IoU(target,prediction,n_classes)
    f1 = get_class_F1(target,prediction,n_classes)

    return acc,iou,f1


def confusion_matrix(y_test,y_pred):

    cm = metrics.confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
    return cm
