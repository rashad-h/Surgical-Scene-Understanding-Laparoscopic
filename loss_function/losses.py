"""
Define our custom loss function.
"""
import numpy as np
from keras import backend as K
import tensorflow as tf
from typing import Callable, Union
import dill



def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_multilabel(y_true, y_pred, i=None, M=10, smooth=1e-5):
#     weights = [0.12643323485389657,
#  10.3935706778605,
#  9.072042082226456,
#  2.1199837539222415,
#  2.567956397643244,
#  11.499961701778012,
#  9.478749671416256,
#  231.69943039928168,
#  8.834319912785878,
#  1.402206162038798]

    if i:
        result = dice_coef(y_true[:,:,:,i], y_pred[:,:,:,i], smooth)
        return result
    else:
        # custom_alpha_2 = [0.1, 0.5, 0.5, 0.25, 0.25, 0.5, 0.4, 0.8, 0.3, 0.25]
        dice = 0
        for index in range(M):
            result = dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], smooth) # * weights[index]
            # result = result * custom_alpha_2[index]
            dice += result
        return dice / M
    

def dice_metric(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred)

def dice_metric_0(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=0)
def dice_metric_1(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=1)
def dice_metric_2(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=2)
def dice_metric_3(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=3)
def dice_metric_4(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=4)
def dice_metric_5(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=5)
def dice_metric_6(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=6)
def dice_metric_7(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=7)
def dice_metric_8(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=8)
def dice_metric_9(y_true, y_pred):
    return dice_coef_multilabel(y_true=y_true, y_pred=y_pred, i=9)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef_multilabel(y_true, y_pred, M=10)



def weighted_categorical_crossentropy(weights):

    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss




def categorical_focal_loss(alpha, gamma=2.):

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_(y_true, y_pred):

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * K.log(y_pred)

        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_
