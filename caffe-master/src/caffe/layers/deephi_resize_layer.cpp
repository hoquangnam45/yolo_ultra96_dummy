#include <vector>

#include "caffe/layers/deephi_resize_layer.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace caffe {

template <typename Dtype>
void DeephiResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  DeephiResizeParameter dr_param = this->layer_param_.deephi_resize_param();

  CHECK((dr_param.has_new_height() && dr_param.has_new_width()) ||
        (!dr_param.has_new_height() && !dr_param.has_new_width()))
      << "new_height and new_width should be set at the same time";
  CHECK((dr_param.has_scale_h() && dr_param.has_scale_w()) ||
        (!dr_param.has_scale_h() && !dr_param.has_scale_w()))
      << "h_scale and w_scale should be set at the same time";

  if (dr_param.has_new_height()) {
    new_height_ = dr_param.new_height();
    new_width_ = dr_param.new_width();
    use_scale_ = false;
  } else {
    scale_h_ = dr_param.scale_h();
    scale_w_ = dr_param.scale_w();
    use_scale_ = true;
  }
}

template <typename Dtype>
void DeephiResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  if (use_scale_) {
    resized_height_ = height_ * scale_h_;
    resized_width_ = width_ * scale_w_;
  } else {
    resized_height_ = new_height_;
    resized_width_ = new_width_;
  }
  top[0]->Reshape(bottom[0]->num(), channels_, resized_height_, resized_width_);
}

template <typename Dtype>
void DeephiResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
 // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {

    const Dtype *batch_data = bottom_data + bottom[0]->offset(n);

      if (this->layer_param_.deephi_resize_param().resize_type() ==
          DeephiResizeParameter_ResizeType_BILINEAR) {
        inter_mode = 0;
      } else {
        inter_mode = 1;
      }
      cv::Mat resize_input(cv::Size(height_,width_),CV_32FC1 , cv::Scalar(0));
      cv::Mat resize_output(cv::Size(resized_height_,resized_width_),CV_32FC1 , cv::Scalar(0));
      for (int c = 0; c < channels_; c++ ){
        for(int h = 0; h < height_; h++){
          for (int w = 0; w < width_; w ++){
            int batch_index = c * height_ * width_ + h * width_ + w;
            resize_input.at<float>(w ,h) = batch_data[batch_index];
          }
        }
        if (!inter_mode){
          cv::resize(resize_input , resize_output , cv::Size(resized_height_ , resized_width_));
        }
        else{
          cv::resize(resize_input , resize_output , cv::Size(resized_height_ , resized_width_),0 , 0 , cv::INTER_NEAREST);
        }
        for(int h = 0; h < resized_height_; h++){
          for (int w = 0; w < resized_width_; w ++){
            int batch_index =  c * resized_height_ * resized_width_ + h * resized_width_ + w;
            top_data[batch_index] = resize_output.at<float>(w ,h);
          }
        }
    }
      top_data += top[0]->offset(1);
  }
}

INSTANTIATE_CLASS(DeephiResizeLayer);
REGISTER_LAYER_CLASS(DeephiResize);

} // namespace caffe
