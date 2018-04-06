#include <opencv2\core.hpp>
#include <opencv2/ml/ml.hpp>

#include "baseDef.h"

float runClassifier(std::string algorithmName, std::string inputFeatures, std::string outputFolder);
float svmClassifier(int numberslabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float knnClassifier(int numberslabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels, int K);
float bayesClassifier(int numberslabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float randomForestClassifier(int numberslabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float logisticRegressionClassifier(int numberslabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels);
float computeAccuracy(cv::Mat predicted, cv::Mat actual);