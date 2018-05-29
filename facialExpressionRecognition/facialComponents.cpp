#include "facialComponents.h"

// Problema cnn con la shape predictor a 68 punti https://github.com/davisking/dlib-models.

void getFace(std::string facialMethod, std::string histType, int version, int imageSourceType, std::string roi, bool facePose, std::string cascadeChose, bool duplicateDataset)
{

	// ***
	//
	// Imposto directory output e input.
	//
	// ***

	std::string inputPath = baseDatabasePath + "/" + nameDataset;
	std::string outputPath;

	// Version è 0 (zero) se sto selezionando la ROI nel dataset.
	if (version == imageVersion::dataset)
	{
		outputPath = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult;
	}
	// Version è 1 se sto selezionando la ROI dell'immagine di testing.
	else
	{
		// Utilizzo un path differente in questo caso perché non voglio "sporcare" la directory del dataset.
		outputPath = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryTest;
	}

	std::vector<std::string> imagePath;
	
	// Se si vuole prendere le immagini da una dadaset.
	if (version == imageVersion::dataset)
	{
		imagePath = getListFile(inputPath, duplicateDataset);
	}
	// Se si vuole dare la singola immagine.
	else
	{
		cv::Mat imageTemp;
		// Se l'immagine che si vuole dare è salvata in una directory.
		if (imageSourceType == imageSourceTestType::file)
		{
			std::string tempPath;
			cout << "Give me image path: " << endl;
			std::cin.clear();
			std::cin.sync();
			std::getline(std::cin, tempPath);
			imageTemp = cv::imread(tempPath, CV_LOAD_IMAGE_COLOR);
		}
		// Se l'immagine vuole che sia presa dalla webcam.
		else
		{
			cv::VideoCapture camera(0);
			if (camera.isOpened())
			{
				camera >> imageTemp;
			}

			camera.release();
		}

		cv::resize(imageTemp, imageTemp, Size(256, 256));

		cv::imwrite(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryTest + "/" + nameImageFileTest, imageTemp);
		imagePath.push_back(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryTest + "/" + nameImageFileTest);
	}

	FileStorage fs(outputPath + "/" + fileList, FileStorage::WRITE);
	int imageSize = 0;
	if (version == imageVersion::dataset)
	{
		imageSize = imagePath.size();
		fs << "number_of_image" << imageSize;

	}

	// ***
	//
	// Fine.
	//
	// ***

	// ***
	//
	// Selezione ROI.
	//
	// ***

	dlib::frontal_face_detector detector;
	dlib::shape_predictor predictor;

	cv::CascadeClassifier faceCascade;
	Ptr<cv::cuda::CascadeClassifier> faceCascadeGpu;

	net_type net;

	cv::dnn::Net netOpenCVDNN;

	if (facialMethod.compare("cnn") && (!roi.compare("roialt") || !roi.compare("chip")))
	{
		dlib::deserialize(baseDatabasePath + "/" + shapePredictorDataName) >> predictor;
	}

	if (!facialMethod.compare("cnn"))
	{
		dlib::deserialize(baseDatabasePath + "/" + cnnFaceDetector) >> net;
		if (!roi.compare("roialt") || !roi.compare("chip"))
		{
			dlib::deserialize(baseDatabasePath + "/" + shapePredictorDataName2) >> predictor;
		}
	}

	if (!facialMethod.compare("dnn"))
	{
		netOpenCVDNN = cv::dnn::readNetFromCaffe(baseDatabasePath + "/" + dnnProtoOpenCV, baseDatabasePath + "/" + dnnModelOpenCV);
	}

	int count = 0;
	int countFacePose = 0;
	int countRoi = 0;

	for (int imageId = 0; imageId < imageSize; ++imageId)
	{

		cv::Mat faceROI;
		cv::Mat image;
		cv::Mat gray;
		cv::Mat output;

		// L'immagine viene caricata con un modello colore BGR.
		image = cv::imread(imagePath[imageId], CV_LOAD_IMAGE_COLOR);

		cv::cvtColor(image, gray, CV_BGR2GRAY);

		if (!facialMethod.compare("cnn"))
		{
			gray.copyTo(output);
		}
		else
		{
			// Aumento dei contrasti nell'immagine.
			if (!histType.compare("clahe"))
			{
				double clipLimit = 4.0f;
				Size tileGridSize(8, 8);
				Ptr<CLAHE> clahe = cv::createCLAHE(2.0, tileGridSize);
				clahe->apply(gray, output);
			}
			else if (!histType.compare("hist"))
			{
				equalizeHist(gray, output);
			}
			else
			{
				gray.copyTo(output);
			}
		}

		dlib::cv_image<uchar> outputDlib(output);

		dlib::full_object_detection shape;

		if (!facialMethod.compare("hog"))
		{
			detector = dlib::get_frontal_face_detector();
			
			std::vector<dlib::rectangle> faces = detector(outputDlib);
			if (faces.size() > 1)
			{
				std::cout << "ERROR: too much faces." << endl;
				return;
			}

			if (faces.size() < 1)
			{
				std::cout << "ERROR: where are faces." << endl;
				return;
			}

			if (facePose)
			{
				shape = predictor(outputDlib, faces[0]);
			}
			else
			{
				faceROI = output(dlibRectangleToOpenCV(faces[0]));
			}
		}
		else if (!facialMethod.compare("cascade"))
		{
			std::vector<cv::Rect> faces;
			if (!checkCudaAvailable())
			{
				if (!cascadeChose.compare("default"))
				{
					faceCascade.load(baseDatabasePath + "/" + cascadeDataName);
				}
				else if (!cascadeChose.compare("alt"))
				{
					faceCascade.load(baseDatabasePath + "/" + cascadeDataName2);
				}
				else if (!cascadeChose.compare("alt2"))
				{
					faceCascade.load(baseDatabasePath + "/" + cascadeDataName3);
				}
				else if (!cascadeChose.compare("lbp"))
				{
					faceCascade.load(baseDatabasePath + "/" + cascadeLbpDataName);
				}
				else if (!cascadeChose.compare("lbp2"))
				{
					faceCascade.load(baseDatabasePath + "/" + cascadeLbpDataName2);
				}
				
				if (faceCascade.empty())
				{
					return;
				}

				faceCascade.detectMultiScale(output, faces, 1.2, 3, 0 | CASCADE_SCALE_IMAGE, cv::Size(50, 50));
			}
			else
			{
				if (!cascadeChose.compare("default"))
				{
					faceCascadeGpu = cv::cuda::CascadeClassifier::create(baseDatabasePath + "/" + cascadeDataNameCuda);
				}
				else if (!cascadeChose.compare("alt"))
				{
					faceCascadeGpu = cv::cuda::CascadeClassifier::create(baseDatabasePath + "/" + cascadeDataName2Cuda);
				}
				else if (!cascadeChose.compare("alt2"))
				{
					faceCascadeGpu = cv::cuda::CascadeClassifier::create(baseDatabasePath + "/" + cascadeDataName3Cuda);
				}
				else if (!cascadeChose.compare("lbp"))
				{
					faceCascadeGpu = cv::cuda::CascadeClassifier::create(baseDatabasePath + "/" + cascadeLbpDataName);
				}
				else if (!cascadeChose.compare("lbp2"))
				{
					faceCascadeGpu = cv::cuda::CascadeClassifier::create(baseDatabasePath + "/" + cascadeLbpDataName2);
				}
				
				if (faceCascadeGpu->empty())
				{
					return;
				}

				cv::cuda::GpuMat outputGpu(output);
				cv::cuda::GpuMat facesGpu;
				faceCascadeGpu->setFindLargestObject(true);
				faceCascadeGpu->setScaleFactor(1.2);
				faceCascadeGpu->setMinNeighbors(3);
				faceCascadeGpu->setMinObjectSize(cv::Size(150, 150));
				faceCascadeGpu->detectMultiScale(outputGpu, facesGpu);
				faceCascadeGpu->convert(facesGpu, faces);
			}

			if (faces.size() > 1)
			{
				std::cout << "ERROR: too much faces." << endl;
				return;
			}

			if (faces.size() < 1)
			{
				std::cout << "ERROR: where are faces." << endl;
				return;
			}

			if (facePose)
			{
				shape = predictor(outputDlib, openCVRectToDlib(faces[0]));
			}
			else
			{
				faceROI = cv::Mat(output, faces[0]);
				//faceROI = output(faces[0]);
			}

		}
		else if (!facialMethod.compare("cnn"))
		{
			try
			{
				dlib::matrix<dlib::rgb_pixel> imageCNN;
				dlib::assign_image(imageCNN, dlib::cv_image<rgb_pixel>(image));
				std::vector<dlib::mmod_rect> dets = net(imageCNN);
				//std::vector<dlib::mmod_rect> dets = net(jitter_image(imageCNN));

				if (facePose)
				{
					shape = predictor(imageCNN, dets[0].rect);
				}
				else
				{
					faceROI = output(dlibRectangleToOpenCV(dets[0].rect));
				}
			}
			catch (exception& e)
			{
				std::cout << e.what() << endl;
			}
		}
		else if (!facialMethod.compare("dnn"))
		{
			cv::Mat temp;
			cv::Mat imageGrayBGR;
			cv::Mat imageFalseBGR[] = { output, output, output };
			cv::merge(imageFalseBGR, 3, temp);
			temp.convertTo(imageGrayBGR, image.type());
			cv::Mat imageDNNBlob = cv::dnn::blobFromImage(imageGrayBGR, 1.0, cv::Size(256, 256), Scalar(104.0, 177.0, 123.0), false, false);
			netOpenCVDNN.setInput(imageDNNBlob, "data");
			cv::Mat detection = netOpenCVDNN.forward("detection_out");
			cv::Mat faces(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
			for (int i = 0; i < faces.rows; i++)
			{
				float confidence = faces.at<float>(i, 2);

				if (confidence > 0.95)
				{
					int left = static_cast<int>(faces.at<float>(i, 3) * imageGrayBGR.cols);
					int top = static_cast<int>(faces.at<float>(i, 4) * imageGrayBGR.rows);
					int right = static_cast<int>(faces.at<float>(i, 5) * imageGrayBGR.cols);
					int bottom = static_cast<int>(faces.at<float>(i, 6) * imageGrayBGR.rows);

					left = std::min(std::max(0, left), imageGrayBGR.cols - 1);
					top = std::min(std::max(0, top), imageGrayBGR.rows - 1);
					right = std::min(std::max(0, right), imageGrayBGR.cols - 1);
					bottom = std::min(std::max(0, bottom), imageGrayBGR.rows - 1);

					cv::Rect faceRect((int)left, (int)top, (int)(right - left), (int)(bottom - top));

					if (facePose)
					{
						shape = predictor(outputDlib, openCVRectToDlib(faceRect));
					}
					else
					{
						//faceROI = cv::Mat(image, faceRect);
						//cv::cvtColor(faceROI, faceROI, COLOR_BGR2GRAY);
						faceROI = cv::Mat(output, faceRect);
					}
					break;
				}
			}
		}
		else
		{
			return;
		}

		// ***
		//
		// Tentativo di prendere la parte di interesse.
		//
		// ***

		if (!roi.compare("roialt"))
		{
			cv::Point centerEyeRight;
			cv::Point centerEyeLeft;
			int widthEyeRight;
			int widthEyeLeft;
			if (!facialMethod.compare("cnn"))
			{
				centerEyeRight = cv::Point(
					(shape.part(1).x() + shape.part(0).x()) / 2,
					(shape.part(1).y() + shape.part(0).y()) / 2);

				centerEyeLeft = cv::Point(
					(shape.part(2).x() + shape.part(3).x()) / 2,
					(shape.part(2).y() + shape.part(3).y()) / 2);

				widthEyeRight = abs(shape.part(1).x() - shape.part(0).x());
				widthEyeLeft = abs(shape.part(2).x() - shape.part(3).x());
			}
			else {
				centerEyeRight = cv::Point(
					(shape.part(42).x() + shape.part(45).x()) / 2,
					(shape.part(42).y() + shape.part(45).y()) / 2);

				centerEyeLeft = cv::Point(
					(shape.part(36).x() + shape.part(39).x()) / 2,
					(shape.part(36).y() + shape.part(39).y()) / 2);

				widthEyeRight = abs(shape.part(42).x() - shape.part(45).x());
				widthEyeLeft = abs(shape.part(36).x() - shape.part(39).x());
			}
			int widthFace = (centerEyeRight.x + widthEyeRight) - (centerEyeLeft.x - widthEyeLeft);
			widthFace *= 1.10;
			int heightFace = widthFace * 1.1;

			faceROI = output(cv::Rect(centerEyeLeft.x - (widthFace / 4), centerEyeLeft.y - (heightFace / 4), widthFace, heightFace));
			
		}
		else if (!roi.compare("chip"))
		{
			dlib::array2d<rgb_pixel> faceChipTemp;
			dlib::extract_image_chip(outputDlib, dlib::get_face_chip_details(shape, dimensionImageOutputResize, 0), faceChipTemp);
			cv::Mat test;
			test = dlib::toMat(faceChipTemp);

			test.convertTo(faceROI, CV_8U);
			cv::cvtColor(faceROI, faceROI, CV_BGR2GRAY);
		}

		if (!roi.compare("roi") || !roi.compare("roialt"))
		{
			cv::resize(faceROI, faceROI, cv::Size(widthImageOutputResize, heightImageOutputResize));
		}

		try
		{
			if (version == imageVersion::dataset)
			{
				std::string filename;
				if (duplicateDataset)
				{
					if (imageId % 2 == 0)
					{
						filename = imagePath[imageId].substr(inputPath.length() + 1, imagePath[imageId].length());
					}
					else
					{
						filename = imagePath[imageId].substr(inputPath.length() + 1 + 10, imagePath[imageId].length());
					}
				}
				else
				{
					filename = imagePath[imageId].substr(inputPath.length() + 1, imagePath[imageId].length());
				}
				
				std::string currentFilename = filename;
				currentFilename.replace(filename.length() - 4, 4, "face.tiff");
		
				cv::imwrite(outputPath + "/" + currentFilename, faceROI);

				fs << "image_" + std::to_string(imageId) + "_face" << outputPath + "/" + currentFilename;
			}
			else
			{
				cv::imwrite(outputPath + "/" + nameImageFileTestRoi, faceROI);
			}
		}
		catch (exception& e)
		{
			std::cout << e.what() << endl;
		}
	}

	std::cout << count << " " << countFacePose << " " << countRoi << endl;

	fs.release();
}

std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel>& img)
{
	// All this function does is make 100 copies of img, all slightly jittered by being
	// zoomed, rotated, and translated a little bit differently. They are also randomly
	// mirrored left to right.
	thread_local dlib::rand rnd;

	std::vector<matrix<rgb_pixel>> crops;
	for (int i = 0; i < 100; ++i)
		crops.push_back(jitter_image(img, rnd));

	return crops;
}

std::vector<std::string> getListFile(std::string directory, bool duplicateDataset)
{
	std::vector<std::string> imagePath;
	DIR *pDIR;
	struct dirent *entry;
	if (pDIR = opendir(directory.c_str()))
	{
		while (entry = readdir(pDIR))
		{
			if (entry->d_type == DT_REG)
			{
				std::string name = entry->d_name;
				std::string::size_type size = name.find(".tiff");
				if (size != std::string::npos)
				{
					imagePath.push_back(directory + "/" + name);
					if (duplicateDataset)
					{
						std::string nameImageDuplicate;
						nameImageDuplicate = duplicateImage(name);
						imagePath.push_back(nameImageDuplicate);
					}
				}
			}
		}
	}
	return imagePath;
}

static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
	return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

bool checkCudaAvailable()
{
	return cv::cuda::getCudaEnabledDeviceCount() && USE_CUDA;
}

std::string duplicateImage(std::string filename)
{
	cv::Mat image = cv::imread(baseDatabasePath + "/" + nameDataset + "/" + filename);
	cv::Mat dst;
	cv::flip(image, dst, 1);
	std::string tempStringName = filename.substr(0, 7);
	std::string tempStringNumber = filename.substr(7, filename.length() - 12);
	int tempId = std::stoi(tempStringNumber);
	tempId += 1;
	tempStringNumber = std::to_string(tempId);
	std::string currentFilename = tempStringName + tempStringNumber + ".tiff";
	cv::imwrite(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryImageDuplicate + "/" + currentFilename, dst);
	return baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryImageDuplicate + "/" + currentFilename;
}