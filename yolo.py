#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
# from keras import backend as K
# import tensorflow.python.keras.backend as K
import tensorflow.compat.v1.keras.backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.5
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None)
        
        self.class_threshold = 0.6
        self.net_h = 416
        self.net_w = 416
        self.obj_thresh = 0.8
        self.nms_thresh = 0.45
        
        self.is_fixed_size = self.model_image_size != (None, None)
        # self.boxes, self.scores, self.classes = self.generate()
        self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors
        #eturn [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        #self.input_image_shape = K.placeholder(shape=(2, ))
        #boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
        #        len(self.class_names), self.input_image_shape,
        #        score_threshold=self.score, iou_threshold=self.iou)
        #return boxes, scores, classes

    def detect_image(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        # self.yolo_model.input = image_data
        # self.input_image_shape = [image.size[1], image.size[0]]

        yolos = self.yolo_model.predict(image_data)

        return_boxs = []

        for i in range(len(yolos)):
            # decode the output of the network
            # we need out_boxes, out_scores, out_classes
            # we are loosing class and score information here!
            
            anchor_idx = 2 - i
            tmpBox = self.decode_netout(yolos[i][0], self.anchors.flatten()[anchor_idx*6:(anchor_idx+1)*6], self.obj_thresh, self.nms_thresh, self.net_h, self.net_w)
            
            for j in range(len(tmpBox)):
                x = (tmpBox[j].x * image.size[0])  
                y = (tmpBox[j].y * image.size[1])  
                w = (tmpBox[j].w * image.size[0])
                h = (tmpBox[j].h * image.size[1])
                if x < 0 :
                    w = w + x
                    x = 0
                if y < 0 :
                    h = h + y
                    y = 0 
                return_boxs.append([x,y,w,h, tmpBox[j].get_label(), tmpBox[j].get_score()])
            
        return return_boxs

    def close_session(self):
        self.sess.close()

    def decode_netout(self, netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5

        boxes = []

        netout[..., :2]  = self._sigmoid(netout[..., :2])
        netout[..., 4:]  = self._sigmoid(netout[..., 4:])
        netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h*grid_w):
            row = i / grid_w
            col = i % grid_w
        
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                #objectness = netout[..., :4]
            
                #if(objectness.all() <= obj_thresh): continue
                if(objectness <= obj_thresh): continue
            
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]

                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]
                
                # if (np.argmax(classes) == 0): continue

                print(self.class_names[np.argmax(classes)] + " - " + str(objectness))

                box = BoundBox(x-w/2, y-h/2, h, w, objectness, classes)
                # box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

                boxes.append(box)

        return boxes

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

class BoundBox:
    def __init__(self, x, y, h, w, objness = None, classes = None):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score