import numpy as np
import cv2
import os
from utility import create_data

from imp import load_source

h,w = 160, 160 # image dimensions
Train = 1 # Number of samples used for training
Val = 100 # Number of samples used for validation during training
np.random.seed(2)
# Take these from train.py
std = 0.09
mean = 0.15
X_test, Y_test = create_data(h, w, 100, size = (15, 40), mode = 'circle')

# Loading the model:
model_base = 'models'
model_id = 'unet_1'
session_id = 'session_1'
model_path = os.path.join(model_base, model_id)
session_path = os.path.join(model_path, 'sessions', session_id)
session = load_source('session',  os.path.join(model_path, '__init__.py'))
Model = session.model
options = session.options

Model.load_weights( os.path.join(session_path, 'weights.hd5') )

for i,x in enumerate(X_test):
    Y = Y_test[i]
    img = ((x.reshape((h, w)) * std) + mean)*255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    R = (Model.predict(x.reshape((1, h, w,1)))*255).reshape((h, w)).astype('uint8')
    Y = (Y*255).reshape((h, w)).astype('uint8')
    mask = np.zeros((h, w,3), dtype = 'uint8')
    mask[:,:,0] = Y
    mask[:,:,2] = R

    cv2.imwrite('vis/test_{}.png'.format(i), mask)