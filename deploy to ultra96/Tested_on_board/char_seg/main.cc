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

// /*
// This function resize non square image to square one (height == width)
// :param img: input image as numpy array
// :return: numpy array
// */
// void square(Mat& img){


//     // image after making height equal to width
//     squared_image = img

//     // Get image height and width
//     h = img.shape[0]
//     w = img.shape[1]

//     // In case height superior than width
//     if h > w:
//         diff = h-w
//         if diff % 2 == 0:
//             x1 = np.zeros(shape=(h, diff//2))
//             x2 = x1
//         else:
//             x1 = np.zeros(shape=(h, diff//2))
//             x2 = np.zeros(shape=(h, (diff//2)+1))

//         squared_image = np.concatenate((x1, img, x2), axis=1)

//     // In case height inferior than width
//     if h < w:
//         diff = w-h
//         if diff % 2 == 0:
//             x1 = np.zeros(shape=(diff//2, w))
//             x2 = x1
//         else:
//             x1 = np.zeros(shape=(diff//2, w))
//             x2 = np.zeros(shape=((diff//2)+1, w))

//         squared_image = np.concatenate((x1, img, x2), axis=0)

//     return squared_image
// }

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
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return (i < j);
}
vector<Mat> plate_segmentation(Mat& BGR_img)){
    //img = cv2.imread(img_file_path)
    Mat imgray;
    cvtColor(RGB_img, imggray, CV_BGR2GRAY);

    int height = RGB_img.size().height;//img.shape[0]
    int width = RGB_img.size().width;//shape[1]
    long area = height * width;

    const double scale1 = 0.001;
    const double scale2 = 0.1;
    double area_condition1 = area * scale1;
    double area_condition2 = area * scale2;
    
    // global thresholding
    //ret1,th1 = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
    Mat th1;
    double ret1 = threshold(imggray, th1, 127, 255, CV_THRESH_BINARY);

    //Otsu's thresholding
    //ret2,th2 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    Mat th2;
    double ret2 = threshold(imggray, th2, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Otsu's thresholding after Gaussian filtering
    Mat blur_img, th3;
    GaussianBlur(imggray, blur_img, Size(5,5), 0);// = cv2.GaussianBlur(imgray,(5,5),0)
    double ret3 = threshold(blur_img, th3, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU); // ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    // sort contours
    //contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    //contours = sorted(contours, key=cv2.contourArea, reverse=True)
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(th3, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), compareContourAreas);

    vector<Mat> cropped;// = []
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)


        if (w * h > area_condition1 and w * h < area_condition2 and w/h > 0.3 and h/w > 0.3):
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
            c = th2[y:y+h,x:x+w]
            c = np.array(c)
            c = cv2.bitwise_not(c)
            c = square(c)
            c = cv2.resize(c,(28,28), interpolation = cv2.INTER_AREA)
            cropped.append(c)
    cv2.imwrite('detection.png', img)
    return cropped;
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
    digits = plate_segmentation(read_img);
    return 0; 
}