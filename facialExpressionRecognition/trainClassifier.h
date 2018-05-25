#include <opencv2/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <dlib/svm_threaded.h>

#include "baseDef.h"

float runClassifier(std::string algorithmName, std::string inputFeatures, std::string outputFolder);
float svmClassifierDlib(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float svmClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float knnClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels, int K);
float bayesClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float randomForestClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float logisticRegressionClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float computeAccuracy(cv::Mat predicted, cv::Mat actual);