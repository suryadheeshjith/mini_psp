import matplotlib.pyplot as plt
#import itertools


def plot_history(history,save_path):
    """
    plots the graph of accuracy and loss metrics
    """
    #ACCURACY PLOT
    plt.plot(history.history['accuracy'],color='orange',label = 'acc')
    plt.plot(history.history['val_accuracy'],color='blue',label = 'val_acc')

    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(save_path+"/"+'accuracy.png')
    plt.show()

    #LOSS PLOT
    plt.plot(history.history['loss'],label = 'loss')
    plt.plot(history.history['val_loss'],label = 'val_loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path+"/"+'loss.png')
    plt.show()


def confusion_matrix_helper(cm, classes,title='Confusion matrix',cmap=plt.cm.Reds):


    cm = cm.astype('float') / cm.sum(axis=0)
    cm = np.round(cm,decimals=10)
    print("Normalized confusion matrix")

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

def plot_confusion_matrix(cm,class_names,save_path):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    confusion_matrix_helper(cm, classes=class_names, title='Normalized confusion matrix')
    plt.savefig(save_path+"/"+'confusion_matrix.png')
    plt.show()
