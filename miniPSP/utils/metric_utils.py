import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#import itertools

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



def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Reds):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=0)
            cm = np.round(cm,decimals=10)
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        print(cm)

        thresh = cm.max(axis=0)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="red" if cm[i, j] >= thresh[j] else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


def confusion_matrix(class_names,y_test,y_pred):


    cm = confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
