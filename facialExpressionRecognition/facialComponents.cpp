#include "facialComponents.h"

void getFace(std::string facialMethod, std::string histType, int version, int imageSourceType, bool roi, bool landmark, int cascadeChose)
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

	dlib::deserialize(baseDatabasePath + "/" + shapePredictorDataName) >> predictor;

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
				std::cout << "ERROR ERROR" << endl;
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
			if (cascadeChose == 0)
			{
				faceCascade.load(baseDatabasePath + "/" + cascadeDataName);
			}
			else if (cascadeChose == 1)
			{
				faceCascade.load(baseDatabasePath + "/" + cascadeDataName2);
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
			faceCascade.detectMultiScale(output, faces, 1.2, 3, 0, cv::Size(50, 50));

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
		else
		{
			return;
		}

		// ***
		//
		// Tentativo di prendere la parte di interesse.
		//
		// ***

		if (!roi)
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

		std::string currentFilename;
		try
		{
			if (version == 0)
			{
				std::string filename = imagePath[imageId].substr(inputPath.length() + 1, imagePath[0].length());
				currentFilename = filename;
				currentFilename.replace(filename.length() - 4, 4, "face.tiff");

				if (roi)
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
				if (roi)
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
