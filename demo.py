#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 0.7
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    webcam_flag = False
    resize_flag = True
    resize_size = (800, 450)

    # some links from earthcam https://github.com/Crazycook/Working/blob/master/Webcams.txt    https://www.vlcm3u.com/web-cam-live/
    # video_url = 'https://videos3.earthcam.com/fecnetwork/lacitytours1.flv/chunklist_w683585821.m3u8' # HOLLYWOOD
    # video_url = 'https://videos3.earthcam.com/fecnetwork/9974.flv/chunklist_w1421640637.m3u8' # NYC
    # video_url = 'https://videos3.earthcam.com/fecnetwork/5775.flv/chunklist_w1803081483.m3u8' # NYC 2
    # video_url = 'http://181.1.29.189:60001/cgi-bin/snapshot.cgi?chn=0&u=admin'
    # video_url = 'https://videos-3.earthcam.com/fecnetwork/15559.flv/chunklist_w573709200.m3u8' # NYC 3
    video_url = 'https://hddn01.skylinewebcams.com/live.m3u8?a=97psdt8nv2hsmclta3nuu4di94'

    if webcam_flag:
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture()
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        video_capture.open(video_url)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        if resize_flag:
            frame = cv2.resize(frame,resize_size, interpolation = cv2.INTER_AREA)

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        if np.array(boxs).size > 0:
            features = encoder(frame,np.array(boxs)[:,0:4].tolist())
        
            class_names = yolo.class_names

            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
        
             #Call the tracker
            tracker.predict()
            tracker.update(detections)
        
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame,  str(track.track_id),(int(bbox[0]), int(bbox[1])-10),0, 5e-3 * 100, (0,0,255),2)

            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                cv2.putText(frame, class_names[int(det.label)] + "(" + str(round(det.score,2)) + ")",(int(bbox[0]), int(bbox[3])),0, 5e-3 * 90, (255,0,0),2)
                #cv2.putText(frame, str(int(bbox[0])) + "-" + str(int(bbox[3])) ,(int(bbox[0]), int(bbox[3])),0, 5e-3 * 90, (0,0,255),2)

        cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())


