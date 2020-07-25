import numpy as np
from sklearn import metrics


def get_iou(target,prediction):

    '''Returns Intersection over Union (IoU).'''

    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# def get_mean_accuracy(target,prediction):
#
#     '''Returns mean accuracy.'''
#
#     return np.mean(target==prediction)
#
# def get_class_accuracies(target,prediction,n_classes):
#
#     '''Returns class accuracies.'''
#
#     assert len(target.shape)==4
#     assert len(prediction.shape)==4
#
#
#     sum =0
#     acc = {}
#     for i in range(n_classes):
#         cur_acc = np.mean(prediction[:,:,:,i]==target[:,:,:,i])
#         sum+=cur_acc
#         acc[i+1] = cur_acc
#     acc['mean'] = sum/n_classes
#     return acc


def get_class_iou(target,prediction,n_classes):

    '''Returns class IoUs.'''

    assert len(target.shape)==4
    assert len(prediction.shape)==4

    sum =0
    IoU = {}
    for i in range(n_classes):
        cur_iou = get_iou(prediction[:,:,:,i],target[:,:,:,i])
        sum+=cur_iou
        IoU[i+1] = cur_iou
    IoU['mean'] = sum/n_classes
    return IoU


def get_class_f1(target,prediction,n_classes):

    '''Returns class F1-scores.'''

    assert len(target.shape)==4
    assert len(prediction.shape)==4

    sum =0
    f1 = {}
    for i in range(n_classes):
        cur_f1 = metrics.f1_score(prediction[:,:,:,i].reshape(-1,1),target[:,:,:,i].reshape(-1,1))
        sum+=cur_f1
        f1[i+1] = cur_f1
    f1['mean'] = sum/n_classes
    return f1


def evaluate(target,prediction,n_classes):

    '''Returns class accuracies, IoUs and F1-scores.'''

    #acc = get_class_accuracies(target,prediction,n_classes)
    iou = get_class_iou(target,prediction,n_classes)
    f1 = get_class_f1(target,prediction,n_classes)

    #return acc,iou,f1
    return iou,f1


def conf_matrix(target,prediction,n_classes):


    '''Returns confusion matrix.'''

    # Need to remove the 0 values in the target mask if any.
    prediction = np.reshape(prediction,(-1,n_classes))
    target = np.reshape(target,(-1,n_classes))

    cm = metrics.confusion_matrix(prediction.argmax(axis=1),target.argmax(axis=1))
    return cm

def eval_conf_matrix(cm,n_classes):

    '''Returns evaluation metrics from confusion matrix.'''

    cm = np.array(cm)

    sum=0;
    total =0;
    prod_acc = [0]*n_classes
    user_acc = [0]*n_classes
    total_pred = [0]*n_classes
    total_test = [0]*n_classes
    gc =0

    for i in range(n_classes):
        for j in range(n_classes):
            total_pred[i]+= cm[i][j]
            total_test[j]+=cm[i][j]
            if i==j:
                sum+=cm[i][j]
            total+=cm[i][j]

    # User and Producer Accuracies
    for i in range(n_classes):
        gc+=total_pred[i]*total_test[i]
        prod_acc[i] = cm[i][i]/total_test[i]
        user_acc[i] = cm[i][i]/total_pred[i]

    # Overall Accuracy
    ovAc = sum/total

    # Kappa coefficient
    kappa = (total*sum - gc)/(total*total - gc)

    # print("Total pred :",total_pred)
    # print("Total target :",total_test)
    # print("Total :",total)
    return ovAc, kappa, prod_acc, user_acc


if __name__=='__main__':

    ######################################################################
    #### TESTING
    ######################################################################
    n_classes = 5

    prediction = np.load('prediction.npy')
    target = np.load('target.npy')
    iou, f1 = evaluate(target,prediction,n_classes)
    print("IoU : ",iou)
    print("F1 : ",f1)

    #cm = conf_matrix(target,prediction,n_classes)

    cm = [  [119397,540,304,12182,7327],
            [243,7169,43,4319,1737],
            [134,0,5776,721,200],
            [827,2,28,7655,811],
            [793,0,57,278,31494]
        ]


    ovAc, kappa, prod_acc, user_acc = eval_conf_matrix(cm,n_classes)
    print("Overall Accuracy : ",ovAc)
    print("Kappa coeff : ",kappa)
    print("Producer Accuracy : ",prod_acc)
    print("User Accuracy : ",user_acc)



    # Kappa checks
    # prediction = np.reshape(prediction,(-1,n_classes))
    # target = np.reshape(target,(-1,n_classes))
    # print("Kappa score : ",metrics.cohen_kappa_score(target.argmax(axis=1),prediction.argmax(axis=1)))
