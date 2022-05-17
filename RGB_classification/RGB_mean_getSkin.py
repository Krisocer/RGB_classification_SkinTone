import os
import cv2
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

ims_path='(ur read file path fill in)' #path
ims_list=os.listdir(ims_path)

def gamma_trans(img,gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

#average of RGB
def NUM_of_SkinLight1(num):
    numbers = {
        num >= 170 : 'white',
        171 > num >= 110 : 'brown',
        110 > num : 'black'
    }
    return numbers

def NUM_of_SkinLight2(num):
    numbers = {
        num >= 196 : 'white',
        196 > num >= 75 : 'brown',
        75 > num : 'black'
    }
    return numbers

def NUM_of_SkinLight3(num):
    numbers = {
        num >= 194 : 'white',
        194 > num >= 109 : 'brown',
        109 > num : 'black'
    }
    return numbers
	
f = open('(saved file path fill in)', 'w', encoding='utf-8', newline="")
csv_write = csv.writer(f)
csv_write.writerow(['image_Num', 'R', 'G' , 'B', 'skin_color(NIS)' , 'skin_color(color Bar scale)' , 'skin_color(AI)'])

#YCrCb(Cr) + OTSY 
''' For gaussian filtering, cr is the source image data to be filtered, 
	(5,5) is the window size of value, and 0 refers to the standard 
	deviation of gaussian function calculated according to the window size
'''
for im_list in ims_list:
	im_R = 0
	im_G = 0
	im_B = 0
	count = 0
	#read the image
	im=cv2.imread(ims_path+im_list, cv2.IMREAD_COLOR) 
	x, y = im.shape[0:2]
	#if x > 500 and y > 500:
		#im = cv2.resize(im, (500,500), interpolation=cv2.INTER_CUBIC)
	img_corrected=gamma_trans(im,0.9)
	if x > 500 and y > 500:
		img_corrected = cv2.resize(img_corrected, (500,500), interpolation=cv2.INTER_CUBIC)
	#covert raw image to YUV
	ycrcb = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2YCrCb) 
	#ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
	(y, cr, cb) = cv2.split(ycrcb)
	# Gaussian filtering is performed on cr channel components
	cr1 = cv2.GaussianBlur(cr, (5, 5), 0) 
	# According to OTSU algorithm, the image threshold is calculated and the image is binarized
	_, skin1 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#open demo
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
	skin1 = cv2.morphologyEx(skin1, cv2.MORPH_OPEN, kernel)
    #extrect value of diffient channel
	width = skin1.shape[1]
	height = skin1.shape[0]
	for i in range(height):
		for j in range(width):
			if skin1[i,j] == 255:
				im_R += img_corrected[i,j,2]
				im_G += img_corrected[i,j,1]
				im_B += img_corrected[i,j,0]
				#im_R += im[i,j,2]
				#im_G += im[i,j,1]
				#im_B += im[i,j,0]
				count += 1
    #count mean for every channel
	im_R_mean = im_R / count
	im_G_mean = im_G / count
	im_B_mean = im_B / count
	#can test the weight here to optimize
	weighted_Mean = (im_R_mean + im_G_mean + im_B_mean) / 3

	csv_write.writerow([im_list, im_R_mean, im_G_mean, im_B_mean, NUM_of_SkinLight1(weighted_Mean), NUM_of_SkinLight2(weighted_Mean), NUM_of_SkinLight3(weighted_Mean)])
f.close()


