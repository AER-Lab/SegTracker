import cv2
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from model import UnetModel
import matplotlib.pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX
import csv




def trackVideo(FrameBatch, videoPath,csvName):

    frameNum = 0

    with open((csvName+'.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "x", "y"])

    uNet = keras.models.load_model('mouseModel.h5')
    cap = cv2.VideoCapture(videoPath)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( totalFrames )

    while(cap.isOpened()):
        Frames = []
        for i in range(0,FrameBatch):
            __ , frame = cap.read()
            frame = cv2.resize(frame,(256,256))
            Frames.append(frame)

        Frames = np.array(Frames).astype(np.float64)
        Masks = uNet.predict(Frames)
        
        with open((csvName+'.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            for mask in Masks:
                #get bianary mask 
                Mask = (mask* 255.0).astype(np.uint8)
                Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)
                ret, imThresh = cv2.threshold(Mask, 10, 255, 0)
                #compute center of mass
                im, contours, hierarchy = cv2.findContours(imThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                largestContour = max(contours, key = cv2.contourArea)
                M = cv2.moments(largestContour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #add to csv
                writer.writerow([frameNum, cX, cY])
                frameNum+=1
        print(frameNum/totalFrames)
videoPath = "EEG_EMG_Nesting-190503-152922_1221_1226_-200119-201430_Cam2.avi"
trackVideo(100, videoPath,"testTrackingCSV")