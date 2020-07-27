import matplotlib.pyplot as plt
import numpy as np
import itertools
import os.path as osp


def plot_history(history,save_path):

    '''Plots and saves accuracy and loss plot using 'history'.'''

    #ACCURACY PLOT
    plt.plot(history.history['acc'],color='orange',label = 'acc')
    plt.plot(history.history['val_acc'],color='blue',label = 'val_acc')

    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(osp.join(save_path,'accuracy.png'))
    plt.show()

    #LOSS PLOT
    plt.plot(history.history['loss'],label = 'loss')
    plt.plot(history.history['val_loss'],label = 'val_loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(osp.join(save_path,'loss.png'))
    plt.show()


def confusion_matrix_helper(cm, classes,title='Confusion matrix',cmap=plt.cm.Reds):


    print("Confusion matrix")

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
        color="white" if cm[i, j] >= thresh[j] else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(cm,class_names,save_path):

    '''Plots and saves the confusion matrix.'''

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    np.set_printoptions(precision=2)
    plt.figure()
    confusion_matrix_helper(cm, classes=class_names, title='Confusion matrix')
    plt.savefig(osp.join(save_path,'confusion_matrix.png'))
    plt.show()