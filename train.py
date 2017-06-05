import numpy as np
import cv2
import os, sys
from utility import create_data
from imp import load_source

h,w = 160, 160 # image dimensions
Train = 1 # Number of samples used for training
Val = 100 # Number of samples used for validation during training
np.random.seed(1)


X_train, Y_train = create_data(h, w, Train, size = (15,40), mode = 'circle')
sys.exit()
# If we set Train = 1, i.e. only 1 training sample, we can speed up training by
# having 100 samples that are all identical
X_train = np.repeat(X_train, 100, 0)
Y_train = np.repeat(Y_train, 100, 0)

X_val, Y_val = create_data(h, w, Val, size = (15,40), mode = 'circle')

# Loading the model:
model_base = 'models'
model_id = 'unet_1'
session_id = 'session_1'
model_path = os.path.join(model_base, model_id)
session_path = os.path.join(model_path, 'sessions', session_id)
session = load_source('session',  os.path.join(model_path, '__init__.py'))
Model = session.model
options = session.options

print (Model.summary())
Model.compile(**options[session_id])
Model.fit(x = X_train, y = Y_train, nb_epoch = 20, batch_size = 1, verbose = True, validation_data = (X_val, Y_val))
Model.save_weights( os.path.join(session_path, 'weights.hd5') )