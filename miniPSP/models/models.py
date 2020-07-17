import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import Model

from utils.model_utils import PSP_NET_helper, crop


def PSP_Net(input_shape = (256, 256, 4), n_classes = 5, optimizer = 'adam', loss = 'categorical_crossentropy'):

    inputs = Input(input_shape)
    out = PSP_NET_helper(inputs,n_classes)
    model = Model(inputs = inputs, outputs = out)
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

    return model


def UNET(input_shape=(256, 256, 4), n_classes = 5, optimizer = 'adam', loss = 'categorical_crossentropy'):

    inputs = Input(input_shape)

    #Encoder
    conv1 = Conv2D(64, 3, activation='relu',padding='same'  )(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu' ,padding='same' )(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up8 = Conv2D(128, 2, activation='relu' ,padding='same')(UpSampling2D(size=(2, 2))(pool2))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu' ,padding='same'  )(merge8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2D(64, 2, activation='relu' ,padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu',padding='same' )(merge9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(n_classes, 1, padding="valid")(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Reshape(
            (input_shape[0]*input_shape[1], n_classes),
            input_shape=(input_shape[0], input_shape[1], n_classes))(conv10)
    conv10 = Activation("softmax")(conv10)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

    return model

def FCN(input_shape=(256, 256, 4), n_classes = 5, optimizer = 'adam', loss = 'categorical_crossentropy'):


    img_input = Input(input_shape)
    x = img_input
    levels = []

    x = (ZeroPadding2D((1, 1)))(x)
    x = (Conv2D(64, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    levels.append(x)

    x = (ZeroPadding2D((1, 1)))(x)
    x = (Conv2D(128, (3, 3),padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    levels.append(x)

    for _ in range(1):
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Conv2D(256, (3, 3), padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((2, 2)))(x)
        levels.append(x)


    [f1, f2, f3] = levels
    o = f3

    o = (Conv2D(4096, (1, 1), activation='relu',padding='same'))(o)
    o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(2, 2), use_bias=False)(o)
    o2 = f2
    o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',))(o2)
    o, o2 = crop(o, o2, img_input)
    o = Add()([o, o2])
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(2, 2), use_bias=False)(o)
    o2 = f1
    o2 = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = Conv2DTranspose(n_classes, kernel_size=(2, 2),  strides=(2, 2), use_bias=False)(o)
    # o = Reshape((256*256, n_classes),input_shape=(256,256, n_classes))(o)
    o = Reshape((input_shape[0]*input_shape[1], n_classes),
            input_shape=(input_shape[0], input_shape[1], n_classes))(o)
    o = Activation("softmax")(o)

    model =  Model(img_input, o)
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

    return model
