/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/

#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>

#include <dnndk/dnndk.h>
#include<opencv2/opencv.hpp>
#include "utils.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

// #define TOLERANCE_FACTOR 10;
// void get_contour_precedence(contour, cols){
//     origin = cv2.boundingRect(contour)
//     return ((origin[1] / tolerance_factor) * tolerance_factor) * cols + origin[0];
// }
/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param img 
 *
 * @return none
 */
vector<double> runTensorflow(DPUTask* task, Mat& img, bool is_gray_img=0, bool invert=0) {
    /* mean values for YOLO-v3 */
    //float mean[3] = {0.0f, 0.0f, 0.0f};

    //int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    //int width = dpuGetInputTensorWidth(task, INPUT_NODE);
    Mat gray_img;
    Size imgsize(28,28);
    resize(img,gray_img,imgsize);

    if (!is_gray_img){    
        cvtColor(gray_img, gray_img, CV_BGR2GRAY);
    }
    
    if (invert) gray_img = 255 - gray_img;

    /* feed input frame into DPU Task */
    int height = dpuGetInputTensorHeight(task, "conv2d_Conv2D");
    int width = dpuGetInputTensorWidth(task, "conv2d_Conv2D");
    int size = dpuGetInputTensorSize(task, "conv2d_Conv2D");
    int8_t* data = dpuGetInputTensorAddress(task, "conv2d_Conv2D");
    float scale = dpuGetInputTensorScale(task, "conv2d_Conv2D");
    //cout << "Input dim: " << height << " " << width << " " << size << ", scale: " << scale << endl;
    //image img_new = load_image_cv(gray_img);
    //image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    const int img_channel = 1;
    for(int b = 0; b < height; ++b) {
        for(int c = 0; c < width; ++c) {
            for(int a = 0; a < img_channel; ++a) {
                bb[b*width*img_channel + c*img_channel + a] = (float) gray_img.data[a*height*width + b*width + c] / 255.;
            }
        }
    }

    for(int i = 0; i < size; ++i) {
        data[i] = int(bb.data()[i]*scale);
        if(data[i] < 0) data[i] = 127;//?
        // cout << (int) data[i] << "," << bb.data()[i] << "||";
    }
    cout << endl;
    // //free_image(img_new);
    // //free_image(img_yolo);

    /* invoke the running of DPU for Tensorflow */
    dpuRunTask(task);


    /*output nodes of YOLO-v3 */
    const vector<string> outputs_node = {"dense_1_MatMul"};

    //vector<vector<float>> boxes;
    for(size_t i = 0; i < outputs_node.size(); i++){
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        //cout << "Output dim: " << height << " " << width << " " << " " << channel << " " << size << ", scale: " << scale << endl;
        vector<float> result(sizeOut);
        //boxes.reserve(sizeOut);

        /* Store every output node results */
        vector<int8_t> nums(sizeOut);
        memcpy(nums.data(), dpuOut, sizeOut);
        vector<double> ret(sizeOut);
        for (int i = 0; i < sizeOut; i++){
            ret[i] = (double) nums[i] * scale;
        }
        return ret;
        // int max = nums[0];
        // int i_max = 0;
        // for (int i = 0; i < sizeOut; i++){
        //     if (nums[i] > max) {
        //         max = nums[i];
        //         i_max = i;
        //     }
        //     cout << (float) nums[i] * scale << ", ";
        // }
        // cout << endl;
        // cout << "Predicted: " << i_max << endl;
        // // for(int a = 0; a < channel; ++a){
        // //     for(int b = 0; b < height; ++b){
        // //         for(int c = 0; c < width; ++c) {
        // //             int offset = b * channel * width + c * channel + a;
        // //             result[a * height * width + b * width + c] = nums[offset] * scale;
        // //             cout << nums[offset] << ",";
        // //         }
        // //     }
        // // }
        // cout << endl;
        // Softmaxing result vector

        // Print result vector
        /* Store the object detection frames as coordinate information  */
        //detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }
    // /* Restore the correct coordinate frame of the original image */
    // correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);

    // /* Apply the computation for NMS */
    // cout << "boxes size: " << boxes.size() << endl;
    // vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    // float h = frame.rows;
    // float w = frame.cols;
    // for(size_t i = 0; i < res.size(); ++i) {
    //     float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
    //     float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
    //     float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
    //     float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;
	
	// cout<<res[i][res[i][4] + 6]<<" ";
	// cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;


    //     if(res[i][res[i][4] + 6] > CONF ) {
    //         int type = res[i][4];

    //         if (type==0) {
    //             rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 0, 255), 1, 1, 0);
    //         }
    //         else if (type==1) {
    //             rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 0), 1, 1, 0);
    //         }
    //         else {
    //             rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 255), 1, 1, 0);
    //         }
    //     }
    // }   
}

/*
This function resize non square image to square one (height == width)
:param img: input image as numpy array
:return: numpy array
*/
Mat square(Mat& img){
    // image after making height equal to width
    Mat squared_img = img;

    // Get image height and width
    int h = img.size().height;//.shape[0]
    int w = img.size().width;//shape[1]

    // In case height superior than width
    if (h > w){
        int diff = h-w;
        Mat x1, x2;
        if (!diff % 2){
            x1 = Mat::zeros(Size(h, diff / 2), CV_8UC1);
            x2 = x1;
            //x1 = np.zeros(shape=(h, diff//2))
            //x2 = x1
        }
        else{
            x1 = Mat::zeros(Size(h, diff / 2), CV_8UC1);
            x2 = Mat::zeros(Size(h, diff / 2 + 1), CV_8UC1);
            //x1 = np.zeros(shape=(h, diff//2))
            //x2 = np.zeros(shape=(h, (diff//2)+1))
        }
        vconcat(x1, img, squared_img);
        vconcat(x2, squared_img, squared_img);
    }

    // In case height inferior than width
    if (h < w){
        int diff = w-h;
        Mat x1, x2;
        if (!diff % 2){
            x1 = Mat::zeros(Size(diff / 2, w), CV_8UC1);
            x2 = x1;
            //x1 = np.zeros(shape=(diff//2, w))
            //x2 = x1
        }
        else{
            x1 = Mat::zeros(Size(diff / 2, w), CV_8UC1);
            x2 = Mat::zeros(Size(diff / 2 + 1, w), CV_8UC1);
            //x1 = np.zeros(shape=(diff//2, w))
            //x2 = np.zeros(shape=((diff//2)+1, w))
        }
        vconcat(x1, img, squared_img);
        vconcat(x2, squared_img, squared_img);
        //squared_image = np.concatenate((x1, img, x2), axis=0)
    }
    return squared_img;
}

// void sort(vector){
//     int sort_flag = True
//     while (sort_flag == True):

//         sort_flag = False
//         for i in range(len(vector) - 1):
//             x_1 = vector[i][0]
//             y_1 = vector[i][1]

//             for j in range(i + 1, len(vector)):

//                 x_2 = vector[j][0]
//                 y_2 = vector[j][1]

//                 if (x_1 >= x_2 and y_2 >= y_1):
//                     tmp = vector[i]
//                     vector[i] = vector[j]
//                     vector[j] = tmp
//                     sort = True

//                 elif (x_1 < x_2 and y_2 > y_1):
//                     tmp = vector[i]
//                     vector[i] = vector[j]
//                     vector[j] = tmp
//                     sort = True
//     return vector
// }

// comparison function object
bool compareContourAreas ( std::vector<cv::Point>& contour1, std::vector<cv::Point>& contour2 ) {
    double i = fabs(contourArea(cv::Mat(contour1)));
    double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}
vector<Mat> plate_segmentation(Mat& BGR_img){
    //img = cv2.imread(img_file_path)
    Mat imgray;
    cvtColor(BGR_img, imgray, CV_BGR2GRAY);

    int height = BGR_img.size().height;//img.shape[0]
    int width = BGR_img.size().width;//shape[1]
    long area = height * width;

    const double scale1 = 0.001;
    const double scale2 = 0.1;
    double area_condition1 = area * scale1;
    double area_condition2 = area * scale2;
    
    // global thresholding
    //ret1,th1 = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
    //Mat th1;
    //double ret1 = threshold(imgray, th1, 127, 255, CV_THRESH_BINARY);

    //Otsu's thresholding
    //ret2,th2 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    Mat th2;
    double ret2 = threshold(imgray, th2, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Otsu's thresholding after Gaussian filtering
    Mat blur_img, th3;
    GaussianBlur(imgray, blur_img, Size(5,5), 0);// = cv2.GaussianBlur(imgray,(5,5),0)
    double ret3 = threshold(blur_img, th3, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU); // ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    // sort contours
    //contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    //contours = sorted(contours, key=cv2.contourArea, reverse=True)
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(th3, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), compareContourAreas);

    vector<Mat> cropped;// = []
    double min_height = 0.35*height, max_height = 0.5*height, min_width = 0.1*width, max_width = 0.15*width;
    for (auto &cnt : contours){
        Rect ROI = boundingRect(cnt);//= (x,y,w,h) = boundingRect(cnt);
        double x = ROI.x, y = ROI.y, w = ROI.width, h = ROI.height; 
        if (h <= max_height && h >= min_height){
        //if (w * h > area_condition1 && w * h < area_condition2 && w/h > 0.3 && h/w > 0.3){
            drawContours(BGR_img, vector<vector<Point> >(1,cnt), -1, Scalar(0, 255, 0), 3);
            rectangle(BGR_img, Point(x,y), Point(x+w, y+h), Scalar(255,0,0), 2);
            Mat c;
            th2(ROI).copyTo(c);
            bitwise_not(c, c);
            //c = square(c);
            resize(c, c, Size(28,28), CV_INTER_AREA);
            cropped.push_back(c); 
            // cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
            // cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
            // c = th2[y:y+h,x:x+w]
            // c = np.array(c)
            // c = cv2.bitwise_not(c)
            // c = square(c)
            // c = cv2.resize(c,(28,28), interpolation = cv2.INTER_AREA)
            // cropped.append(c)
        }
    }
    imwrite("detection.png", BGR_img);
    return cropped;
}

int reduce(vector<double>& ret){
    double max = ret[0];
    int i_max = 0;
    for (int i = 0; i < ret.size(); i++){
        if (ret[i] > max) {
            max = ret[i];
            i_max = i;
        }
    }
    return i_max;
}

int main(int argc, char **argv){
    if (argc != 2){
        cout << "Please use it like this: ./char_seg <image_path>";
        exit(-1);
    }
    char* img_file_path = argv[1];
    Mat read_img = imread(img_file_path);
    if(!read_img.data){
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    vector<Mat> digits = plate_segmentation(read_img);
    //namedWindow("Display window", WINDOW_AUTOSIZE);
    dpuOpen();
    DPUKernel *kernel = dpuLoadKernel("char_regconition");//char_regconition");
    DPUTask* task = dpuCreateTask(kernel, 0);
    for (auto& digit: digits){
        static int i = 0;
        string name = "out" + to_string(i) + ".jpg";
        vector <double> ret = runTensorflow(task, digit, 1);
        cout << "Raw return vector output of image " + name << ": ";
        for (int i = 0; i < ret.size(); i++){
            if (i) cout << ", "; 
            cout << ret[i];
        }
        cout << endl;
        cout << "Prediction: " << reduce(ret) << endl;
        i++;
        imwrite(name, digit);
    }

    dpuDestroyTask(task);
    /* Destroy DPU Kernels & free resources */
    dpuDestroyKernel(kernel);

    /* Dettach from DPU driver & free resources */
    dpuClose();
    return 0; 
}