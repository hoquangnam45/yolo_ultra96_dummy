#!/bin/bash

python ../yolo_convert.py 0_model_darknet/yolov3-tiny.cfg  0_model_darknet/yolov3-tiny.weights 1_model_caffe/v3-tiny.prototxt 1_model_caffe/v3-caffe.caffemodel
