# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
import os
import csv

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

# Python program to check if rectangles overlap
class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
  
# Returns true if two rectangles(l1, r1)  
# and (l2, r2) overlap 
def doOverlap(l1, r1, l2, r2):

    # If one rectangle is on left side of other 
	if(l1.x > r2.x or l2.x > r1.x):
		return False
  
    # If one rectangle is above other 
	if(l1.y < r2.y or l2.y < r1.y):
		return False
  
	return True

def rectArea(w,h):
	return w*h


directory = './toMeasure/'
files = [f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f)) and f.split('.')[-1]=='jpg')]

print(files)
print('Enter scale:')
scale_length = input()

for filename in files:
	# print('Enter file name:')
	# filename = input()
	print('File: {}'.format(filename))

	# load the image, convert it to grayscale, and blur it slightly
	image = cv2.imread('./toMeasure/{}'.format(filename))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21,21), 0)
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges

	cv2.imshow('image',gray)
	cv2.waitKey(0)

	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	cv2.imshow('image',edged)
	cv2.waitKey(0)

	remove_edge = cv2.dilate(edged, None, iterations=7)
	remove_edge = cv2.erode(remove_edge, None, iterations=8)
	remove_edge = cv2.dilate(remove_edge, None, iterations=1)

	cv2.imshow('image',remove_edge)
	cv2.waitKey(0)

	contours, hier = cv2.findContours(remove_edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt)>100:
			cv2.drawContours(edged,[cnt],0,0,-1)

	cv2.imshow('image',edged)
	cv2.waitKey(0)

	contours.sort(key=lambda x:get_contour_precedence(x, image.shape[1]))
	
	contours = [cnt for cnt in contours if cv2.contourArea(cnt)>2500]
	ref_contour = contours[0]
	print(cv2.contourArea(ref_contour))
	(ref_x,ref_y,ref_w,ref_h) = cv2.boundingRect(ref_contour)

	cv2.rectangle(image,(ref_x,ref_y),
				(ref_x+ref_w,ref_y+ref_h),
				(0,255,0),1)

	scale = (float(scale_length)/ref_w)

	edged = cv2.dilate(edged, None, iterations=15)

	contours, hier = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		if cv2.contourArea(cnt)<1500:
			cv2.drawContours(edged,[cnt],0,0,-1)

	cv2.imshow('image',edged)
	cv2.waitKey(0)

	edged = (medial_axis(edged).astype(np.uint8))*255

	cv2.imshow('image',edged)
	cv2.waitKey(0)

	contours, hier = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = [cnt for cnt in contours if cv2.contourArea(cnt)>1000]
	rects = [cv2.boundingRect(cnt) for cnt in contours]

	len_cnts = len(contours)

	rects_toKeep = []

	for rect1 in range(0,len_cnts):
		(x1,y1,w1,h1) = rects[rect1]
		l1 = Point(x1,y1)
		r1 = Point(x1+w1,y1-h1)
		count_overlaps = []
		for rect2 in range(0,len_cnts):
			(x2,y2,w2,h2) = rects[rect2]
			l2 = Point(x2,y2)
			r2 = Point(x2+w2,y2-h2)
			if(doOverlap(l1, r1, l2, r2)):
				if rectArea(w1,h1)>rectArea(w2,h2):
					count_overlaps.append(rect2)
					if len(count_overlaps)>2:
						areas = []
						for i in count_overlaps:
							(x2,y2,w2,h2) = rects[i]
							area_overlap = areas.append(abs(rectArea(w1,h1)-rectArea(w2,h2)))
						rect2 = count_overlaps[areas.index(max(areas))]
					rects_toKeep.append(rects[rect2])
				else:
					continue
			else:
				continue

	rects_toKeep = list(set(rects_toKeep))

	with open('eggs_dimensions.csv', 'a') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow([filename,scale_length])

		if len(rects_toKeep) == 0:
			print("Couldn't find any eggs :(")
			filewriter.writerow([None, None])
		else:
			for rect in rects_toKeep:

				cv2.rectangle(image,(rect[0],rect[1]),
						(rect[0]+rect[2],rect[1]+rect[3]),
						(0,0,255),1)

				i = rects_toKeep.index(rect)
				width = '{}mm'.format(str(rect[2]*scale)[:6])
				height = '{}mm'.format(str(rect[3]*scale)[:6])

				filewriter.writerow([i,width, height])

				cv2.putText(image, '{}-{}mm,{}mm'.format(i,str(rect[2]*scale)[:5],str(rect[3]*scale)[:5]),
					(int(rect[0]-10), int(rect[1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
					0.45, (119, 119, 119), 1)
			
				print('{}-{},{}'.format(i,width,height))

	cv2.imwrite('./Measured/{}'.format(filename), image) 
	cv2.imshow('image',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()