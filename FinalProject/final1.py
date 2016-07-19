# Author: Tongtong Zhao & Saozhong Han

from myro import *
import numpy as np
import cv2

def intercept(avg_px, picWidth):
    distance = picWidth/2 - avg_px
    degree = int((distance/float(picWidth))*60)
    turnBy(degree,"deg")
    forward(2,3)

def detectBlobs(im,params,picNum):
    k=40
    while(1):
        mask = cv2.inRange(im, np.array([0,k,k],np.uint8), np.array([10, 255, 255],np.uint8))
        threshold = cv2.bitwise_and(mask, mask, mask =mask)
        # Set up the detector with given parameters
        detector = cv2.SimpleBlobDetector(params)
        keypoints = detector.detect(threshold)
        if len(keypoints) == 0:
            ret,threshold = cv2.threshold(threshold,1,255,cv2.THRESH_BINARY_INV)
        cv2.imwrite("threshold_%02d.jpg" % picNum, threshold)
        keypoints = detector.detect(threshold)
        if(len(keypoints)!= 0 or k > 100):
            return keypoints
        else:
            k += 10

def searchIntruder():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold =0
    params.maxThreshold = 255
    # Select darker blobs
    params.filterByColor = True
    params.blobColor =0

    params.filterByCircularity = True
    params.minCircularity = 0
    params.maxCircularity = 1

    params.filterByInertia= True
    params.minInertiaRatio = 0
    params.maxInertiaRatio = 1

    params.filterByConvexity = True
    params.minConvexity = 0
    params.maxConvexity = 1
    
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 99999999999999

    findFlag = False
    degree = 0
    picNum = 0
    findingAngle= 60
    turningDirection = 1 # 1: Turn left; -1: Turn right
    avg_px = 0
    setPicSize("small")
    autoCamera()

    while not findFlag:
    	if (degree <= 360):
    		degree = degree + findingAngle
    	else:
    		degree = degree - 360
        picture = takePicture()
        savePicture(picture,"test_%02d.jpg" % picNum)
        im = cv2.imread('test_%02d.jpg' % picNum)
        hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        keypoints = detectBlobs(hsv,params,picNum)
        picNum += 1
        if len(keypoints) != 0:
            print "Found!"
            findFlag = True
            color=(0,255,0)
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('Output.jpg',im_with_keypoints)
            avg_px = 0
            for k in range(len(keypoints)):
                avg_px +=keypoints[k].pt[0]
            avg_px = avg_px/(len(keypoints))
            intercept(avg_px,picture.getWidth())
            findFlag = False
            degree = 0
        else:
            print "Not Found!"
            turnBy(findingAngle * turningDirection, "deg")
            wait(0.5)

def main():
	init("/dev/tty.Fluke2-0B55-Fluke2")
    keypoints = searchIntruder()
    
main()
