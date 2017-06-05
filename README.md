# Segmentation-Experiment
Using Keras (U-Net architecture) to segment shapes on noise.

**General**:

I created this experiment to toy around with some convolutional networks for image segmentation, mostly U-Net architecture (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
For this I create a noise-image and at a random location within that noise-image I put a circle of random size. 
This circle can be visually different from the noise in different ways: taking constant values or a little bit stronger noise.

The reason I put this Code online was mostly to discuss possible interpretation of the interesting observation: Training the network (~13 million parameters)
on only 1 training sample (that means: only 1 image with a circle of random size at a random location) is already sufficient to make
the network converge on a validation/test set (i.e. on noise-images with circles of other sizes and other locations the network was not trained on).

This result surprised me because due to the size of the network I expected the network to simply 'memorize' (i.e. overfit)
on the one training sample that I provided. Instead it seems to find the simplest hypothesis (which is not memorization) 
that is sufficient to fit the training sample, and this hypothesis seems to be the one we actually seek.

The following image shows the training sample that is produced by 'train.py' and the one image I used to train the network with: 

![Training Sample](http://i.imgur.com/i9OQfNb.png)

The network is then trained to segment the circle.


**Code**: 

Just run train.py to load the u-net model and train the model. The function 'create_data' will create the training and validation
data set and save the images to the 'output' directory (I like visualizing my data). 'X_' and 'Y_' denote the noise-images and the ground-truth mask respectively.

The test.py file tests the trained network on some newly created data by showing ground-truth and the predicted mask in one image which is saved 
in the 'vis' directory.





**My current setup**:
- Windows 10 64 bit with an Nvidia 980 TI
- Python 3.6
- opencv 3.2.0
- Tensorflow 1.2.0-rc1
- Keras 2.0.4
