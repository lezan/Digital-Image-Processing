#include "facialComponents.h"

// Problema dnn con la shape predictor a 68 punti https://github.com/davisking/dlib-models.

void getFace(std::string facialMethod, std::string histType, int version, int imageSourceType, std::string roi, bool facePose, std::string cascadeChose)
{

	// ***
	//
	// Imposto directory output e input.
	//
	// ***

	std::string inputPath = baseDatabasePath + "/" + nameDataset;
	std::string outputPath = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult;

	// Version è 0 (zero) se sto selezionando la ROI nel dataset.
	if (version == 0)
	{
		outputPath = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult;
	}
	// Version è 1 se sto selezionando la ROI dell'immagine di testing.
	else
	{
		// Utilizzo un path differente in questo caso perché non voglio "sporcare" la directory del dataset.
		outputPath = baseDatabasePath + "/" + nameDataset + "/" + "temp";
	}

	std::vector<std::string> imagePath;
	
	// Se si vuole prendere le immagini da una dadaset.
	if (version == 0)
	{
		imagePath = getListFile(inputPath);
	}
	// Se si vuole dare la singola immagine.
	else
	{
		// Se l'immagine che si vuole dare è salvata in una directory.
		if (imageSourceType == 0)
		{
			std::string tempPath;
			cout << "Give me image path: " << endl;
			std::cin.clear();
			std::cin.sync();
			std::getline(std::cin, tempPath);
			imagePath.push_back(tempPath);
		}
		// Se l'immagine vuole che sia presa dalla webcam.
		else
		{
			cv::VideoCapture camera(0);
			cv::Mat imageTempCamera;
			if (camera.isOpened())
			{
				camera >> imageTempCamera;
			}

			cv::resize(imageTempCamera, imageTempCamera, Size(256, 256));

			cv::imwrite(baseDatabasePath + "/" + nameDataset + "/" + "temp" + "/" + "imageTempCamera.tiff", imageTempCamera);
			imagePath.push_back(baseDatabasePath + "/" + nameDataset + "/" + "temp" + "/" + "imageTempCamera.tiff");
			camera.release();
		}
	}

	int imageSize = imagePath.size();

	FileStorage fs(outputPath + "/" + fileList, FileStorage::WRITE);
	if (version == 0)
	{
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

	if (facialMethod.compare("dnn") && (!roi.compare("roialt") || !roi.compare("chip")))
	{
		dlib::deserialize(baseDatabasePath + "/" + shapePredictorDataName) >> predictor;
	}

	if (!facialMethod.compare("dnn"))
	{
		dlib::deserialize(baseDatabasePath + "/" + dnnFaceDetector) >> net;
		if (!roi.compare("roialt") || !roi.compare("chip"))
		{
			dlib::deserialize(baseDatabasePath + "/" + shapePredictorDataName2) >> predictor;
		}
	}

	cv::Mat faceROI;
	//cv::Mat faceROIAlt;
	//cv::Mat faceChip;

	for (int imageId = 0; imageId < imageSize; ++imageId)
	{
		cv::Mat image;
		cv::Mat gray;
		cv::Mat output;

		// L'immagine viene caricata con un modello colore BGR.
		image = cv::imread(imagePath[imageId], CV_LOAD_IMAGE_COLOR);

		// Se l'input è l'immagine di testing ed è in una directory.
		if (version == 1 && imageSourceType == 0)
		{
			cv::resize(image, image, Size(256, 256));
		}

		cv::cvtColor(image, gray, CV_BGR2GRAY);

		if (!facialMethod.compare("dnn"))
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
		else if (!facialMethod.compare("dnn"))
		{
			try
			{
				dlib::matrix<dlib::rgb_pixel> imageDNN;
				dlib::assign_image(imageDNN, dlib::cv_image<rgb_pixel>(image));
				std::vector<dlib::mmod_rect> dets = net(imageDNN);
				//std::vector<dlib::mmod_rect> dets = net(jitter_image(imageDNN));

				if (facePose)
				{
					shape = predictor(imageDNN, dets[0].rect);
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
			if (!facialMethod.compare("dnn"))
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

		std::string currentFilename;
		try
		{
			if (version == 0)
			{
				std::string filename = imagePath[imageId].substr(inputPath.length() + 1, imagePath[0].length());
				currentFilename = filename;
				currentFilename.replace(filename.length() - 4, 4, "face.tiff");
		
				cv::imwrite(outputPath + "/" + currentFilename, faceROI);

				fs << "image_" + std::to_string(imageId) + "_face" << outputPath + "/" + currentFilename;
			}
			else
			{
				cv::imwrite(outputPath + "/" + "imageTempROI.tiff", faceROI);
			}
		}
		catch (exception& e)
		{
			std::cout << e.what() << endl;
		}
	}

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

std::vector<std::string> getListFile(std::string directory)
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