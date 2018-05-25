#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include "BowKMajorityTrainer.h"
#include "baseDef.h"

void featureExtraction(std::string featuresExtractionAlgorithm);
cv::Mat extractFeaturesFromSingleImage(std::string featuresExtractionAlgorithm);
cv::Mat runExtractFeature(cv::Mat image, std::string featureName);
cv::Mat extractFeaturesKaze(cv::Mat image);
cv::Mat extractFeaturesSift(cv::Mat image);
cv::Mat extractFeaturesSurfDlib(cv::Mat image);
cv::Mat extractFeaturesSurf(cv::Mat image);
cv::Mat extractFeaturesDaisy(cv::Mat image);
cv::Mat extractFeaturesBrisk(cv::Mat image);
cv::Mat extractFeaturesOrb(cv::Mat image);