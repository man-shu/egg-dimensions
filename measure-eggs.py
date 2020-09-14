# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize
import os
import csv

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def rectArea(x):
	return x[0]*x[1]

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = (dist_from_center <= radius).astype(np.uint8)
    return mask

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
	(ref_x,ref_y,ref_w,ref_h) = cv2.boundingRect(ref_contour)
	cv2.rectangle(image,(ref_x,ref_y),
				(ref_x+ref_w,ref_y+ref_h),
				(0,255,0),1)
	scale = (float(scale_length)/ref_w)

	kernel = create_circular_mask(38,38)
	edged = cv2.dilate(edged, kernel, iterations=1)

	contours, hier = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		if cv2.contourArea(cnt)<1500:
			cv2.drawContours(edged,[cnt],0,0,-1)

	cv2.imshow('image',edged)
	cv2.waitKey(0)

	# edged = np.int0(edged/255)
	edged = (medial_axis(edged).astype(np.uint8))*255

	cv2.imshow('image',edged)
	cv2.waitKey(0)

	contours, hier = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	contours = [cnt for cnt in contours if cv2.contourArea(cnt)>6000]

	tilted_rects = [cv2.minAreaRect(cnt) for cnt in contours]
	# print(tilted_rects)
	dims = [rect[1] for rect in tilted_rects]
	boxes = [np.int0(cv2.boxPoints(rect)) for rect in tilted_rects]

	boxes = sorted(boxes, key=cv2.contourArea)
	dims = sorted(dims, key=rectArea)
	len_boxes = len(boxes)

	with open('eggs_dimensions.csv', 'a') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow([filename,scale_length])
		if len_boxes == 0:
			print("Couldn't find any eggs :(")
			filewriter.writerow([None, None])
		else:
			# if len_boxes%2 == 0:
			# 	boxes = boxes[0:int(len_boxes/2)]
			# 	dims = dims[0:int(len_boxes/2)]
			# else:
			# 	boxes = boxes[0:int(len_boxes/2+1)]
			# 	dims = dims[0:int(len_boxes/2+1)]

			for dim in dims:
				i = dims.index(dim)

				cv2.drawContours(image,[boxes[i]],0,(0,0,255),1)

				width = '{}mm'.format(str(dim[0]*scale)[:6])
				height = '{}mm'.format(str(dim[1]*scale)[:6])
				
				if width>height:
					temp_w = height
					height = width
					width = temp_w

				box = perspective.order_points(boxes[i])
				(tl, tr, br, bl) = box
				(tltrX, tltrY) = midpoint(tl, tr)
				(tlblX, tlblY) = midpoint(tl, bl)

				cv2.putText(image, '{}-{}'.format(i,width),
					(int(tltrX-30), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX,
					0.45, (0, 255, 0), 1)
				cv2.putText(image, '{}-{}'.format(i,height),
					(int(tlblX), int(tlblY)), cv2.FONT_HERSHEY_SIMPLEX,
					0.45, (0, 0, 255), 1)

				filewriter.writerow([i,width, height])
				print('{}-{},{}'.format(i,width,height))

	cv2.imwrite('./Measured/{}'.format(filename), image) 
	cv2.imshow('image',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()