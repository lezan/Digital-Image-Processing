#ifndef defbase
const std::string baseDatabasePath = "../../Database";
//const std::string baseDatabasePath = "Database";
const std::string shapePredictorDataName = "shape_predictor_68_face_landmarks.dat";
const std::string shapePredictorDataName2 = "shape_predictor_5_face_landmarks.dat";
const std::string cnnFaceDetector = "mmod_human_face_detector.dat";
const std::string cascadeDataName = "haarcascade_frontalface_default.xml";
const std::string cascadeDataName2 = "haarcascade_frontalface_alt.xml";
const std::string cascadeDataName3 = "haarcascade_frontalface_alt2.xml";
const std::string cascadeLbpDataName = "lbpcascade_frontalface.xml";
const std::string cascadeLbpDataName2 = "lbpcascade_frontalface_improved.xml";
const std::string fileList = "list.yml";
const std::string nameDataset = "jaffe";
const std::string nameDirectoryResult = "result";
const std::string nameFileFeatures = "Features.yml";
const std::string nameOutputFileAccuracyResult = "result.txt";
const std::string nameSVMModelTrained = "svmModelTrained.xml";
const std::string nameKnnModelTrained = "knnModelTrained.xml";
const std::string nameBayesModelTrained = "bayesModelTrained.xml";
const std::string nameRandomForestModelTrained = "randomForestModelTrained.xml";
const std::string nameLogisticRegressionModelTrained = "logisticRegressionModelTrained.xml";
const std::string nameDictionary = "dictionary.yml";
const std::string namePca = "pca.yml";
const int widthImageOutputResize = 160;
const int heightImageOutputResize = 160;

#include <string>
#include <iostream>

#include "utilityFunction.h"
#include "dirent.h"

using namespace std;
using namespace cv;

#define defbase
#endif

#ifndef FACIAL_COMPONENTS_DO
#define FACIAL_COMPONENTS_DO 1
#endif

#ifndef FEATURES_COMPONENTS_DO
#define FEATURES_COMPONENTS_DO 1
#endif

#ifndef TRAIN_CLASSIFIER_DO
#define TRAIN_CLASSIFIER_DO 1
#endif

#ifndef DO_PREDICT
#define DO_PREDICT 1
#endif

#ifndef MASS_TEST
#define MASS_TEST 0
#endif