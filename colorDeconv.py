import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hed
import numpy as np
from skimage.exposure import rescale_intensity
import skimage.color
from PIL import Image
from glob import glob
import os
import copy
from skimage import measure
from scipy import signal
from time import time

from tqdm import tqdm

def semi_transparent(img,sh=0.31):
    imgb = skimage.color.rgb2hed(img)
    imgb = rescale_intensity(imgb, out_range=(0, 1))
    
    mask = imgb[:,:,2].copy()
    mask[mask>=sh] = 255
    mask[mask<sh] = 0

    img = np.array(img)
    trans_img = img.copy()

    w,h = mask.shape
    

    for row in range(w):
        for col in range(h):
            for channel in range(3):
                if mask[row][col] == 255:
                    trans_img[row][col][channel] = min(1.2*trans_img[row][col][channel],255)
                else:
                    trans_img[row][col][channel] /= transparent_strength

    return trans_img

    
def colordeconv(img,sh=0.31):            
    imgb = skimage.color.rgb2hed(img)
    imgb = rescale_intensity(imgb, out_range=(0, 1))
    
    d = imgb[:,:,2].copy()
    d[d>=sh] = 255
    d[d<sh] = 0

    return d

def denoise(img):
    copy_img = img.copy()

    #kernel = np.ones((3,3),np.uint8)
    #3*3 Gassian filter
    x, y = np.mgrid[-1:2, -1:2]
    kernel = np.exp(-(x**2+y**2))
    kernel = kernel / kernel.sum()

    copy_img = cv2.dilate(copy_img,kernel,iterations = 2)
    copy_img = cv2.erode(copy_img,kernel,iterations = 2)

    #copy_img = cv2.erode(copy_img,kernel,iterations = 2)
    #copy_img = cv2.dilate(copy_img,kernel,iterations = 2)


    return copy_img

def find_connected_component(img):
    
    rt_img = []
    for threshold in area_threshold: 
        grid = img.copy()

        labels = measure.label(grid, connectivity=1)
        for region in measure.regionprops(labels):
            if region.area <= threshold:
                (min_row, min_col, max_row, max_col) = region.bbox
                grid[min_row:max_row,min_col:max_col] = 0


        rt_img.append(grid)

    return rt_img

def plot():
    plt.subplot(231).set_title('Original')
    plt.imshow(trans_img)
    
    plt.subplot(232).set_title('DeConv')
    plt.imshow(imghed)

    plt.subplot(233).set_title('Erode/Dilate')
    plt.imshow(denoised_img)

    for i,threshold in enumerate(area_threshold):
        plt.subplot(2,3,i+4).set_title('Areas >= '+str(threshold)+" Pixels")
        plt.imshow(Remove_SmallArea_imgs[i])
    
    plt.show()
    


if __name__ == "__main__":
    
    
    area_threshold = [50,100,150]
    transparent_strength = 2.5
    
    img_path = './data/29_90641_56413_0600_KI-67.jpg'
    
    imgrgb = Image.open(img_path).convert('RGB')

    trans_img = semi_transparent(imgrgb)
    
    start = time()
    
    imghed = colordeconv(imgrgb)

    denoised_img = denoise(imghed)

    Remove_SmallArea_imgs = find_connected_component(denoised_img)

    print("Execute Time : %f sec"%(time()-start))
    
    plot()
