import numpy as np
from keras.models import Model
from keras.layers import Activation, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Reshape, Permute, Concatenate, concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.optimizers import Optimizer

img_rows = 160
img_cols = 160

smooth = 1.

def mask_binary_regression_error(y_true, y_pred):
    return K.mean(K.log(1 + K.exp(-y_true*y_pred)))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

inputs = Input((img_rows, img_cols, 1)) # 160 x 160
conv1 = Conv2D(32, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(32, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
# pool1 = Dropout(0.15)(pool1)

conv2 = Conv2D(64, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(64, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
# pool2 = Dropout(0.25)(pool2)

conv3 = Conv2D(128, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(128, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
# pool3 = Dropout(0.4)(pool3)

conv4 = Conv2D(256, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(256, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
# pool4 = Dropout(0.5)(pool4)

conv5 = Conv2D(512, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(512, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5) # 5x5
# pool5 = Dropout(0.5)(pool5)

conv6 = Conv2D(512, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool5)
conv6 = Conv2D(512, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv6)


up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5],  axis = 3)
# up7 = Dropout(0.5)(up7)
conv7 = Conv2D(256, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(up7)
conv7 = Conv2D(256, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv7)


up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4],  axis = 3)
# up8 = Dropout(0.4)(up8)
conv8 = Conv2D(128, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(up8)
conv8 = Conv2D(128, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv8)

up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3],  axis = 3)
# up9 = Dropout(0.25)(up9)
conv9 = Conv2D(64, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(up9)
conv9 = Conv2D(64, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)

up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2],  axis = 3)
# up10 = Dropout(0.15)(up10)
conv10 = Conv2D(32, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(up10)
conv10 = Conv2D(32, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv10)

up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1],  axis = 3)
# up11 = Dropout(0.15)(up11)
conv11 = Conv2D(16, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(up11)
conv11 = Conv2D(16, kernel_size = (3,3), strides = 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv11)

conv12 = Conv2D(1, kernel_size = (1,1), strides = 1, activation='sigmoid', kernel_initializer = 'he_normal')(conv11) # n x 2 x 160 x 160
# conv12 = Conv2D(2, 1, 1, kernel_initializer = 'he_normal')(conv11) # n x 2 x 160 x 160

# r = Reshape((2, 160*160))(conv12)
# r = Permute((2,1))(r)
# r = Activation('softmax')(r)
# r = Activation('sigmoid')(conv12)


model = Model(input=inputs, output=conv12)

options = {
    'session_1':
        {
            'loss' : dice_coef_loss,
            # 'loss' : 'binary_crossentropy',
            'optimizer': SGD(lr = 0.001),
            'metrics' : [dice_coef]
        }

}

