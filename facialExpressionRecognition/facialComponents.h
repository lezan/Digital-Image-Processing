#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/cudaobjdetect.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>

#include "baseDef.h"

using namespace dlib;

void getFace(std::string method, std::string histType, int version, int imageSourceType, std::string roi, bool facePose, std::string cascadeChose, bool dupllicateDataset);
std::vector<std::string> getListFile(std::string directory, bool duplicateDataset);
static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
static dlib::rectangle openCVRectToDlib(cv::Rect r);
bool checkCudaAvailable();
std::string duplicateImage(std::string filename);

/** CNN DLIB**/

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

/** END **/