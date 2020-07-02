import cv2
import numpy as np
from matplotlib import pyplot as plt

def sketch1(image):
	img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	img_gray_blur= cv2.GaussianBlur(img_gray, (5,5),0)
	
	canny_edges= cv2.Canny(img_gray_blur, 10, 70)

	
	ret, mask = cv2.threshold(canny_edges,70,255,cv2.THRESH_BINARY)
	
	return mask
	
	
def sketch2(image):
	img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	img_gray_blur= cv2.GaussianBlur(img_gray, (5,5),0)
	
	sobel_x=cv2.Sobel(img_gray_blur,cv2.CV_64F,1,0,ksize=5)

	
	ret, mask = cv2.threshold(sobel_x,70,255,cv2.THRESH_BINARY)
	
	return mask
	

def sketch3(image):
	img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	img_gray_blur= cv2.GaussianBlur(img_gray, (5,5),0)
	
	sobel_y=cv2.Sobel(img_gray_blur,cv2.CV_64F,0,1,ksize=5)

	
	ret, mask = cv2.threshold(sobel_y,70,255,cv2.THRESH_BINARY)
	
	return mask
	
	
	
	
#if you want to implement it on an image	
	
rgb = cv2.imread("atharva.png",)

# remove noise
image = cv2.GaussianBlur(rgb,(3,3),0)

cv2.imshow('Canny',sketch1(image))
cv2.imshow('sobel Y',sketch2(image))
cv2.imshow('Sobel X',sketch3(image))


#if you want to implement in real time video capturing
	
'''
cap = cv2.VideoCapture(0)

while True:
	ret,frame = cap.read()
	cv2.imshow('Canny',sketch1(frame))
	cv2.imshow('sobel Y',sketch2(frame))
	cv2.imshow('Sobel X',sketch3(frame))
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
'''
