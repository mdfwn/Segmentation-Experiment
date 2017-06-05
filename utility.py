import numpy as np
import cv2

def create_data(h, w, N, size, mode = 'circle'):
    '''
    Function to create training samples - N images of size h x w which are
    saved in the directory 'output'.

    h,w : height and width (image dimensions)
    N : number of images
    mode : 'circle' will create circles, 'rectangle' will create rectangles
    size: tuple containing lower bound and upper bound for the size of the circle/rectangle
    fill : how to fill the circles in the image; 'noise' will fill
    '''

    X = np.zeros((N, h, w), dtype = 'uint8')
    Y = np.zeros((N, h, w), dtype = 'uint8')
    for i, M in enumerate(Y):

        if mode == 'circle':
            r = np.random.randint(size[0], size[1])
            # chose position of circle such that all of the circle will be within the image:
            y, x = np.random.randint(2 * r, h - 2 * r), np.random.randint(2 * r, w - 2 * r)
            cv2.circle(M, (x,y), r, 255, -1)

        elif mode == 'rectangle':
            height, width = np.random.randint(size[0], size[1], 2)
            # height = width # Using squares
            y = np.random.randint(0, h-height-1)
            x = np.random.randint(0, w-width-1)
            M[y:y+height, x:x+width] = 255

        Y[i,:,:] = M

        noise1 = (np.random.randint(0, 80, size = (h,w))).astype('uint8')
        noise2 = (np.random.randint(0, 80, size = (h,w))).astype('uint8')

        # cv2.circle(noise1, (x,y), r, np.mean(M) ,-1) # Constant value for circles?
        noise1 = cv2.add(noise1,(M/255).astype('uint8') * noise2)

        X[i, :, :] = noise1
        cv2.imwrite('output/X_{}.png'.format(i+1), noise1)
        cv2.imwrite('output/Y_{}.png'.format(i+1), M)
    Y = Y.astype('float32').reshape((N,h,w,1)) / 255.0
    X = X.astype('float32').reshape((N,h,w,1)) / 255.0
    mean, std = np.mean(X), np.std(X)
    print (mean, std)
    X = (X - mean)/std

    return X, Y