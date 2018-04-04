#include "facialExpression.h"

/*

argv[1] : facial method
	-> default : cascade;
	-> cascade : cascade;
	-> hog : hog;
argv[2] : histogram type
	-> default : null;
	-> hist : histogram;
	-> clahe : clahe;
argv[3] : if duplication of dataset
	-> default: no;
	-> 0 : no;
	-> 1 : yes;

*/

int main(int argc, char* argv[])
{

	std::string facialMethod; // cascade o hog
	std::string histType; //hist o clahe
	bool roi = true;
	bool landmark = true;
	int cascadeChose = 0; // 0 -> default, 1 -> alt, 2 -> alt2
	std::string tempString;

	std::cout <<
		"param 1 : facial method" << endl <<
			"\t -> default : cascade;" << endl <<
			"\t -> cascade : cascade;" << endl <<
			"\t -> cnn : cnn " << endl <<
            "\t -> hog : hog;" << endl <<
        "param 2 : histogram type" << endl <<
            "\t -> default : null;" << endl <<
            "\t -> hist : histogram;" << endl <<
            "\t -> clahe : clahe;" << endl <<
        "param 3 : roi or roialt (only with landmark)" << endl <<
            "\t -> default : roi;" << endl <<
            "\t -> roi : roi;" << endl <<
            "\t -> roialt : roialt;" << endl <<
        "param 4 : if cascade want landmark" << endl <<
            "\t -> default : yes;" << endl <<
            "\t -> no : no, without;" << endl <<
            "\t -> yes : yes, with landmark;" << endl <<
        "param 5 : if cascade want default cascade, alt or alt2" << endl <<
            "\t -> default : default cascade;" << endl <<
            "\t -> alt : alt cascade;" << endl <<
            "\t -> alt2 : alt2 cascade;" << endl <<
        "param 6 : if duplication of dataset" << endl <<
            "\t -> default : no;" << endl <<
            "\t -> 0 : no;" << endl <<
            "\t -> 1 : yes;" << endl;

	if (argc > 1)
	{
		tempString = argv[1];

		if (!tempString.compare("cascade"))
		{
			facialMethod = "cascade";
		}

		if (!tempString.compare("hog"))
		{
			facialMethod = "hog";
		}

		if (!tempString.compare("cnn"))
		{
			facialMethod = "cnn";
		}

		if(!tempString.compare("default"))
		{
			facialMethod = "cascade";
		}
	}

	if (argc > 2)
	{
		tempString = argv[2];

		if (!tempString.compare("hist"))
		{
			histType = "hist";
		}

		if (!tempString.compare("clahe"))
		{
			histType = "clahe";
		}

		if(!tempString.compare("default"))
		{
			histType = "";
		}
	}

	if (argc > 3)
	{
		tempString = argv[3];

		if (!tempString.compare("roi"))
		{
			roi = true;
		}

		if (!tempString.compare("roialt"))
		{
			roi = false;
		}

		if (!tempString.compare("default"))
		{
			roi = true;
		}
	}

	if (argc > 4 && !facialMethod.compare("cascade"))
	{
		tempString = argv[4];
		if (!tempString.compare("yes"))
		{
			landmark = true;
		}

		if (!tempString.compare("no"))
		{
			landmark = false;
		}

		if (!tempString.compare("no") && roi == false)
		{
			landmark = true;
			std::cout << "You can not use roialt without landmark. Force to use lankmark" << endl;
		}

		if (!tempString.compare("default"))
		{
			landmark = true;
		}
	}

	if (argc > 5 && !facialMethod.compare("cascade"))
	{
		tempString = argv[5];

		if (!tempString.compare("default"))
		{
			cascadeChose = 0;
		}

		if (!tempString.compare("alt"))
		{
			cascadeChose = 1;
		}

		if (!tempString.compare("alt2"))
		{
			cascadeChose = 2;
		}
	}

	if (argc > 6)
	{
		if ((int)argv[6] == 1)
		{
			duplicateDatabase(baseDatabasePath + "/" + nameDataset);
		}
	}

	std::cout << "Facial chose: " << facialMethod << endl;
	std::cout << "Hist chose: " << histType << endl;
	std::cout << "ROI chose: " << roi << endl;
	std::cout << "Landmark chose: " << landmark << endl;
	std::cout << "XML cascade chose: " << cascadeChose << endl;

	if (MASS_TEST)
	{
		std::map<std::string, std::map<std::string, std::vector<float> > > result;

		for (int k = 0; k < 2; ++k) // Metodo: hog o cascade. 0 -> hog, 1 -> cascade
		{
			for (int j = 0; j < 2; ++j) // Tipo: hist o clahe. 0 -> hist, 1 -> clahe
			{
				for (int i = 0; i < 2; ++i)
				{
					if (FACIAL_COMPONENTS_DO)
					{
						if (k == 0)
						{
							facialMethod = "hog";
						}
						else if (k == 1)
						{
							facialMethod = "cascade";
						}
						else
						{
							return -1;
						}
						if (j == 0)
						{
							histType = "hist";
						}
						else if (j == 1)
						{
							histType = "clahe";
						}
						else
						{
							return -2;
						}

						long long startFacialComponents = milliseconds_now();
						getFace(facialMethod, histType, 0, 0, roi, landmark, cascadeChose);
						long long elapsedFacialComponents = milliseconds_now() - startFacialComponents;
						std::cout << "Time elapsed for facial components: " << elapsedFacialComponents / 1000 << "s." << endl;

					}

					std::string featureAlgorithm = "sift";

					if (FEATURES_COMPONENTS_DO)
					{
						long long startFeatureExtraction = milliseconds_now();
						featureExtraction(featureAlgorithm);
						long long elapsedFeatureExtraction = milliseconds_now() - startFeatureExtraction;
						std::cout << "Time elapsed for features extraction: " << elapsedFeatureExtraction / 1000 << "s." << endl;
					}

					std::string algorithmName = "svm";
					float temp = runClassifier(algorithmName, baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + featureAlgorithm + nameFileFeatures, baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult);
					result[facialMethod][histType].push_back(temp);

					deleteFileIntoDirectory(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult);
				}
			}
		}

		std::string outputFile = baseDatabasePath + "/" + nameFileFeatures + "/" + "/" + nameOutputFileAccuracyResult;
		ofstream ou;
		ou.open(outputFile);

		std::vector<float>::iterator items;
		std::map<std::string, std::vector<float > >::iterator types;
		std::map < std::string, std::map < std::string, std::vector<float > > > ::iterator groups;

		for (groups = result.begin(); groups != result.end(); groups++) {
			ou << "Method: " << (*groups).first << endl;
			for (types = (*groups).second.begin(); types != (*groups).second.end(); types++) {
				ou << "\tType: " << (*types).first << endl;
				for (items = (*types).second.begin(); items != (*types).second.end(); items++) {
					ou << "\t\tAccuracy: " << *items << endl;
				}
			}
		}

		ou.close();
	}
	else
	{
		if (FACIAL_COMPONENTS_DO)
		{
			long long startFacialComponents = milliseconds_now();
			getFace(facialMethod, histType, 0, 0, roi, landmark, cascadeChose);
			long long elapsedFacialComponents = milliseconds_now() - startFacialComponents;
			std::cout << "Time elapsed for facial components: " << elapsedFacialComponents / 1000 << "s." << endl;
		}

		std::string featuresExtractor = "sift";

		if (FEATURES_COMPONENTS_DO)
		{
			long long startFeatureExtraction = milliseconds_now();
			featureExtraction(featuresExtractor);
			long long elapsedFeatureExtraction = milliseconds_now() - startFeatureExtraction;
			std::cout << "Time elapsed for features extraction: " << elapsedFeatureExtraction / 1000 << "s." << endl;
		}

		std::string algorithmName = "svm";

		if (TRAIN_CLASSIFIER_DO)
		{
			float temp1 = runClassifier(algorithmName, baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + featuresExtractor + nameFileFeatures, baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult);
		}

		if (DO_PREDICT)
		{
			cv::Ptr<cv::ml::SVM> svm;
			svm = cv::ml::StatModel::load<ml::SVM>(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + nameSVMModelTrained);

			int sourceImage = 0; // 0 -> image; 1 -> camera;

			std::string facialMethodInRun;
			std::string histTypeInRun;
			std::string featuresExtractorInRun = "sift";

			std::string c;
			while (true)
			{
				if (sourceImage == 0) // static image
				{
					getFace(facialMethod, histType, 1, 0, roi, landmark, cascadeChose);
				}
				else if (sourceImage == 1) // camera image
				{
					getFace(facialMethod, histType, 1, 1, roi, landmark, cascadeChose);
				}
				else
				{
					return -4;
				}

				cv::Mat feature = extractFeaturesFromSingleImage(featuresExtractorInRun);
				float labelPredicted = svm->predict(feature);
				std::cout << "Label predicted is: " << labelPredicted << endl;

				std::cout << "Go on? Y(yes) or N(no)" << endl;
				std::cin.clear();
				std::cin.sync();
				std::getline(std::cin, c);
				if (c == "N")
				{
					break;
				}
				else if (c == "Y")
				{
					std::cout << "Let's go!" << endl;

					deleteFileIntoDirectory(baseDatabasePath + "/" + nameDataset + "/temp");
				}
				else
				{
					break;
				}
			}
		}
	}

	cv::destroyAllWindows();
	return 0;
}

void duplicateDatabase(std::string directory)
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
	for (int i = 0; i < imagePath.size(); ++i)
	{
		cv::Mat image = imread(imagePath[i]);
		cv::Mat dst;
		cv::flip(image, dst, 1);
		std::string repeat = imagePath[i].substr(directory.length() + 1 + 5, 1);
		std::string filename = imagePath[i].insert(directory.length() + 1 + 5 + 1, repeat);
		cv::imwrite(filename, dst);
	}
}

void deleteFileIntoDirectory(std::string path)
{
	DIR *directory = opendir(path.c_str());
	struct dirent *entry;

	while (entry = readdir(directory))
	{
		if (entry->d_type == DT_REG)
		{
			std::string name = entry->d_name;
			std::string::size_type fileTiff = name.find(".tiff");
			std::string::size_type fileYml = name.find(".yml");
			if (fileTiff != std::string::npos)
			{
				remove((path + "/" + name).c_str());
			}
			if (fileYml != std::string::npos)
			{
				remove((path + "/" + name).c_str());
			}
		}
	}
	closedir(directory);
}