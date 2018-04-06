#include "facialComponents.h"

struct LANDMARKPOSITION
{
	double x;
	double y;
};

LANDMARKPOSITION shape68[] =
{
	{0.0792396913815, 0.339223741112}, {0.0829219487236, 0.456955367943},
	{0.0967927109165, 0.575648016728}, {0.122141515615, 0.691921601066},
	{0.168687863544, 0.800341263616}, {0.239789390707, 0.895732504778},
	{0.325662452515, 0.977068762493}, {0.422318282013, 1.04329000149},
	{0.531777802068, 1.06080371126}, {0.641296298053, 1.03981924107},
	{0.738105872266, 0.972268833998}, {0.824444363295, 0.889624082279},
	{0.894792677532, 0.792494155836}, {0.939395486253, 0.681546643421},
	{0.96111933829, 0.562238253072}, {0.970579841181, 0.441758925744},
	{0.971193274221, 0.322118743967}, {0.163846223133, 0.249151738053},
	{0.21780354657, 0.204255863861}, {0.291299351124, 0.192367318323},
	{0.367460241458, 0.203582210627}, {0.4392945113, 0.233135599851},
	{0.586445962425, 0.228141644834}, {0.660152671635, 0.195923841854},
	{0.737466449096, 0.182360984545}, {0.813236546239, 0.192828009114},
	{0.8707571886, 0.235293377042}, {0.51534533827, 0.31863546193},
	{0.516221448289, 0.396200446263}, {0.517118861835, 0.473797687758},
	{0.51816430343, 0.553157797772}, {0.433701156035, 0.604054457668},
	{0.475501237769, 0.62076344024}, {0.520712933176, 0.634268222208},
	{0.565874114041, 0.618796581487}, {0.607054002672, 0.60157671656},
	{0.252418718401, 0.331052263829}, {0.298663015648, 0.302646354002},
	{0.355749724218, 0.303020650651}, {0.403718978315, 0.33867711083},
	{0.352507175597, 0.349987615384}, {0.296791759886, 0.350478978225},
	{0.631326076346, 0.334136672344}, {0.679073381078, 0.29645404267},
	{0.73597236153, 0.294721285802}, {0.782865376271, 0.321305281656},
	{0.740312274764, 0.341849376713}, {0.68499850091, 0.343734332172},
	{0.353167761422, 0.746189164237}, {0.414587777921, 0.719053835073},
	{0.477677654595, 0.706835892494}, {0.522732900812, 0.717092275768},
	{0.569832064287, 0.705414478982}, {0.635195811927, 0.71565572516},
	{0.69951672331, 0.739419187253}, {0.639447159575, 0.805236879972},
	{0.576410514055, 0.835436670169}, {0.525398405766, 0.841706377792},
	{0.47641545769, 0.837505914975}, {0.41379548902, 0.810045601727},
	{0.380084785646, 0.749979603086}, {0.477955996282, 0.74513234612},
	{0.523389793327, 0.748924302636}, {0.571057789237, 0.74332894691},
	{0.672409137852, 0.744177032192}, {0.572539621444, 0.776609286626},
	{0.5240106503, 0.783370783245}, {0.477561227414, 0.778476346951}
};

const int INNER_EYES_AND_BOTTOM_LIP[] = { 39, 42, 57 };

void getFace(std::string facialMethod, std::string histType, int version, int imageSourceType, std::string roi, bool landmark, std::string cascadeChose)
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

	net_type net;

	if (!facialMethod.compare("hog") || (!facialMethod.compare("cascade") && !roi.compare("roialt")))
	{
		dlib::deserialize(baseDatabasePath + "/" + shapePredictorDataName) >> predictor;
	}

	if (!facialMethod.compare("cnn"))
	{
		dlib::deserialize(baseDatabasePath + "/" + shapePredictorDataName2) >> predictor;
		dlib::deserialize(baseDatabasePath + "/" + cnnFaceDetector) >> net;
	}

	cv::Mat faceROI;
	cv::Mat faceROIAlt;

	for (int imageId = 0; imageId < imageSize; ++imageId)
	{
		cv::Mat image;
		cv::Mat gray;
		cv::Mat output;

		image = cv::imread(imagePath[imageId], CV_LOAD_IMAGE_COLOR);

		// Se l'input è l'immagine di testing ed è in una directory.
		if (version == 1 && imageSourceType == 0)
		{
			cv::resize(image, image, Size(256, 256));
		}

		cv::cvtColor(image, gray, CV_BGR2GRAY);

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

		dlib::cv_image<uchar> cimg(output);

		dlib::full_object_detection shape;

		if (!facialMethod.compare("hog"))
		{
			detector = dlib::get_frontal_face_detector();
			
			std::vector<dlib::rectangle> faces = detector(cimg);
			if (faces.size() > 1)
			{
				std::cout << "ERROR: too much faces." << endl;
			}

			if (faces.size() < 1)
			{
				std::cout << "ERROR: where are faces." << endl;
			}
			shape = predictor(cimg, faces[0]);

            try
            {
				faceROI = output(dlibRectangleToOpenCV(faces[0]));
            }
            catch (exception& e)
            {
                std::cout << e.what() << endl;
            }

		}
		else if (!facialMethod.compare("cascade"))
		{
			if (!cascadeChose.compare("defaul"))
			{
				faceCascade.load(baseDatabasePath + "/" + cascadeDataName);
			}
			else if (!cascadeChose.compare("alt"))
			{
				faceCascade.load(baseDatabasePath + "/" + cascadeDataName2);
			}
			else if (!cascadeChose.compare("lbp"))
			{
				faceCascade.load(baseDatabasePath + "/" + cascadeLbpDataName);
			}
			else if (!cascadeChose.compare("lbp2"))
			{
				faceCascade.load(baseDatabasePath + "/" + cascadeLbpDataName2);
			}
			else
			{
				faceCascade.load(baseDatabasePath + "/" + cascadeDataName3);
			}

			if (faceCascade.empty())
			{
				return;
			}

			std::vector<cv::Rect> faces;
			faceCascade.detectMultiScale(output, faces, 1.2, 3, 0 | CASCADE_SCALE_IMAGE, cv::Size(50, 50));

			if (faces.size() > 1)
			{
				std::cout << "ERROR: too much faces." << endl;
			}

			if (faces.size() < 1)
			{
				std::cout << "ERROR: where are faces." << endl;
			}

			int bestIndex = 0;
			int maxWidth = 0;
			for (unsigned int i = 0; i < faces.size(); ++i) 
			{
				if (faces[i].width > maxWidth) 
				{
					bestIndex = i;
					maxWidth = faces[i].width;
				}
			}

			if (landmark)
			{
				shape = predictor(cimg, openCVRectToDlib(faces[bestIndex]));

				try
				{
					faceROI = output(faces[bestIndex]);
				}
				catch (exception& e)
				{
					std::cout << e.what() << endl;
				}
			}
			else
			{
				faceROI = cv::Mat(output, faces[bestIndex]);
			}

		}
		else if (!facialMethod.compare("cnn"))
		{
			try
			{
				dlib::matrix<dlib::rgb_pixel> img;
				dlib::assign_image(img, dlib::cv_image<uchar>(output));
				std::vector<dlib::mmod_rect> dets = net(img);

				shape = predictor(cimg, dets[0].rect);
				faceROI = output(dlibRectangleToOpenCV(dets[0].rect));
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
			cv::Point centerEyeRight = cv::Point(
				(shape.part(42).x() + shape.part(45).x()) / 2,
				(shape.part(42).y() + shape.part(45).y()) / 2);

			cv::Point centerEyeLeft = cv::Point(
				(shape.part(36).x() + shape.part(39).x()) / 2,
				(shape.part(36).y() + shape.part(39).y()) / 2);

			int widthEyeRight = abs(shape.part(42).x() - shape.part(45).x());
			int widthEyeLeft = abs(shape.part(36).x() - shape.part(39).x());

			int widthFace = (centerEyeRight.x + widthEyeRight) - (centerEyeLeft.x - widthEyeLeft);
			widthFace *= 1.10;
			int heightFace = widthFace * 1.1;

			faceROIAlt = output(cv::Rect(centerEyeLeft.x - (widthFace / 4), centerEyeLeft.y - (heightFace / 4), widthFace, heightFace));

			cv::resize(faceROIAlt, faceROIAlt, cv::Size(widthImageOutputResize, heightImageOutputResize));
		}
		else
		{
			cv::resize(faceROI, faceROI, cv::Size(widthImageOutputResize, heightImageOutputResize));
		}

		/*LANDMARKPOSITION shapeMin, shapeMax;
		shapeMin.x = 1000.0f;
		shapeMin.y = 1000.0f;
		shapeMax.x = -1.0f;
		shapeMax.y = -1.0f;

		for (int i = 0; i < 68; ++i)
		{
			if (shape68[i].x > shapeMax.x)
			{
				shapeMax.x = shape68[i].x;
			}

			if (shape68[i].y > shapeMax.y)
			{
				shapeMax.y = shape68[i].y;
			}

			if (shape68[i].x < shapeMin.x)
			{
				shapeMin.x = shape68[i].x;
			}

			if (shape68[i].y < shapeMin.y)
			{
				shapeMin.y = shape68[i].y;
			}
		}

		LANDMARKPOSITION minMaxShape[68];

		for (int i = 0; i < 68; ++i)
		{
			minMaxShape[i].x = (shape68[i].x - shapeMin.x) / (shapeMax.x - shapeMin.x);
			minMaxShape[i].y = (shape68[i].y - shapeMin.y) / (shapeMax.y - shapeMin.y);
		}*/
		
		std::string currentFilename;
		try
		{
			if (version == 0)
			{
				std::string filename = imagePath[imageId].substr(inputPath.length() + 1, imagePath[0].length());
				currentFilename = filename;
				currentFilename.replace(filename.length() - 4, 4, "face.tiff");

				if (!roi.compare("roi"))
				{
					cv::imwrite(outputPath + "/" + currentFilename, faceROI);

				}
				else
				{
					cv::imwrite(outputPath + "/" + currentFilename, faceROIAlt);

				}

				fs << "image_" + std::to_string(imageId) + "_face" << outputPath + "/" + currentFilename;
			}
			else
			{
				if (!roi.compare("roi"))
				{
					cv::imwrite(outputPath + "/" + "imageTempROI.tiff", faceROI);
				}
				else
				{
					cv::imwrite(outputPath + "/" + "imageTempROI.tiff", faceROIAlt);

				}
			}
		}
		catch (exception& e)
		{
			std::cout << e.what() << endl;
		}
	}

	fs.release();
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