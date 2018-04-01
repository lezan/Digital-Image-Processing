#include <map>
#include <fstream>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

#include "facialComponents.h"
#include "featureExtraction.h"
#include "trainClassifier.h"

void duplicateDatabase(std::string directory);
void deleteFileIntoDirectory(std::string path);