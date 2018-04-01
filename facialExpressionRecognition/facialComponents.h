#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <opencv2/core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include "baseDef.h"

using namespace dlib;

//int facialComponents(cv::Mat image);
void getFace(std::string method, std::string histType, int version, int imageSourceType, bool roi, bool landmark, int cascadeChose);
std::vector<std::string> getListFile(std::string directory);
static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
static dlib::rectangle openCVRectToDlib(cv::Rect r);