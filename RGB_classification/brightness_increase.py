import os
import cv2
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

ims_path='(ur read file path fill in)/' #path
ims_list=os.listdir(ims_path)

def gamma_trans(img,gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)
	
for im_list in ims_list:
	
	im=cv2.imread(ims_path+im_list, cv2.IMREAD_COLOR) 
	img_corrected=gamma_trans(im,0.7)
	cv2.imwrite(ims_path+im_list,img_corrected)



