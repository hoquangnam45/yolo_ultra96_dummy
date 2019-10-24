#!/bin/bash

#yolov2 test on darknet

#./darknet/darknet detector valid cfg/voc.data yolov2.cfg yolov2.weights

#yolov2 test on caffe

#./caffe-master/build/examples/yolo/yolov2_detect.bin \
#          yolov2.prototxt yolov2.caffemodel image.txt \
#          -classes 20 \
#          -out_file result.txt \
#          -confidence_threshold 0.005


#yolov3 test on darknet
#./darkent/darknet detector valid cfg/voc.data yolov3.cfg yolov3.weights

#yolov3 test on caffe
./caffe-master/build/examples/yolo/yolov3_detect.bin \
          yolov3.prototxt yolov3.caffemodel image.txt \
          -out_file result.txt \
          -confidence_threshold 0.005 \
          -classes 80 \
          -anchorCnt 3
