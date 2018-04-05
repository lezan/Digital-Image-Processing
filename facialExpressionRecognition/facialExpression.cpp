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
	std::string roi;
	bool landmark;
	std::string cascadeChose; // 0 -> default, 1 -> alt, 2 -> alt2
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
        "param 5 : if cascade want default cascade, alt, alt2, lbp or lbp2" << endl <<
            "\t -> default : default cascade;" << endl <<
            "\t -> alt : alt cascade;" << endl <<
            "\t -> alt2 : alt2 cascade;" << endl <<
			"\t -> lbp : lbp cascade;" << endl <<
			"\t -> lbp2 : lbp2 cascade;" << endl <<
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
		else if (!tempString.compare("hog"))
		{
			facialMethod = "hog";
		}
		else if (!tempString.compare("cnn"))
		{
			facialMethod = "cnn";
		}
		else if(!tempString.compare("default"))
		{
			facialMethod = "cascade";
		}
		else
		{
			facialMethod = "cascade";
			std::cout << "Error facialMethod: put a default (cascade)." << endl;
		}
	}

	if (argc > 2)
	{
		tempString = argv[2];

		if (!tempString.compare("hist"))
		{
			histType = "hist";
		}
		else if (!tempString.compare("clahe"))
		{
			histType = "clahe";
		}
		else if(!tempString.compare("default"))
		{
			histType = "";
		}
		else
		{
			histType = "";
			std::cout << "Error histType: put a default (null)." << endl;
		}
	}

	if (argc > 3)
	{
		tempString = argv[3];

		if (!tempString.compare("roi"))
		{
			roi = "roi";
		}
		else if (!tempString.compare("roialt"))
		{
			if (!facialMethod.compare("cnn"))
			{
				std::cout << "Error: Can not use cnn with roialt at the moment. Force to use roi." << endl;
				roi = "roi";
			}
			else
			{
				roi = "roialt";
			}
		}
		else if (!tempString.compare("default"))
		{
			roi = "roi";
		}
		else
		{
			roi = "roi";
			std::cout << "Error ROI: put a defualt (roi)." << endl;
		}
	}

	if (argc > 4 && !facialMethod.compare("cascade"))
	{
		tempString = argv[4];
		if (!tempString.compare("yes"))
		{
			landmark = true;
		}
		else if (!tempString.compare("no"))
		{
			if (!roi.compare("roialt"))
			{
				landmark = true;
				std::cout << "You can not use roialt without landmark. Force to use lankmark." << endl;
			}
			else
			{
				landmark = false;
			}
		}
		else if (!tempString.compare("default"))
		{
			landmark = true;
		}
		else
		{
			landmark = true;
			std::cout << "Error landmkar: put a default (yes)." << endl;
		}
	}

	if (argc > 5 && !facialMethod.compare("cascade"))
	{
		tempString = argv[5];

		if (!tempString.compare("default"))
		{
			cascadeChose = "default";
		}
		else if (!tempString.compare("alt"))
		{
			cascadeChose = "alt";
		}
		else if (!tempString.compare("alt2"))
		{
			cascadeChose = "alt2";
		}
		else if (!tempString.compare("lbp"))
		{
			cascadeChose = "lbp";
		}
		else if (!tempString.compare("lbp2"))
		{
			cascadeChose = "lbp2";
		}
		else
		{
			cascadeChose = "default";
			std::cout << "Error cascadeChose: put a default (default)." << endl;
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
			std::string roiInRun;
			bool landmarkInRun;
			std::string cascadeChoseInRun;
			std::string tempStringInRun;

			std::string featuresExtractorInRun = "sift";

			std::string c;
			while (true)
			{
				std::cout << endl << "Give me facial method (cascade, hog, cnn)" << endl;
				std::cin.clear();
				std::cin.sync();
				std::getline(std::cin, tempStringInRun);
				if (!tempStringInRun.compare("cascade"))
				{
					facialMethodInRun = "cascade";
				}
				else if (!tempStringInRun.compare("hog"))
				{
					facialMethodInRun = "hog";
				}
				else if (!tempStringInRun.compare("cnn"))
				{
					facialMethodInRun = "cnn";
				}
				else
				{
					facialMethodInRun = "cascade";
					std::cout << "Error facialMethod: put a default (cascade).";
				}

				std::cout << "Give me hist type (default, hist, clahe)" << endl;
				std::cin.clear();
				std::cin.sync();
				std::getline(std::cin, tempStringInRun);
				if (!tempStringInRun.compare("hist"))
				{
					histTypeInRun = "hist";
				}
				else if (!tempStringInRun.compare("clahe"))
				{
					histTypeInRun = "clahe";
				}
				else if (!tempStringInRun.compare("default"))
				{
					histTypeInRun = "default";
				}
				else
				{
					histTypeInRun = "default";
					std::cout << "Error histType: put a default (null).";
				}

				std::cout << "Give me roi (roi, roialt, default)" << endl;
				std::cin.clear();
				std::cin.sync();
				std::getline(std::cin, tempStringInRun);
				if (!tempStringInRun.compare("roi"))
				{
					roiInRun = "roi";
				}
				else if(!tempStringInRun.compare("roialt"))
				{
					if (!facialMethodInRun.compare("cnn"))
					{
						std::cout << "Error: Can not use cnn with roialt at the moment. Force to use roi." << endl;
						roiInRun = "roi";
					}
					else
					{
						roiInRun = "roialt";
					}
				}
				else if (!tempString.compare("default"))
				{
					roiInRun = "roi";
				}
				else
				{
					roiInRun = "roi";
					std::cout << "Error ROI: put a default (roi)." << endl;
				}

				std::cout << "Give me landmark (yes, no, default)" << endl;
				std::cin.clear();
				std::cin.sync();
				std::getline(std::cin, tempStringInRun);
				if (!tempStringInRun.compare("yes"))
				{
					landmarkInRun = true;
				}
				else if(!tempStringInRun.compare("no"))
				{
					if (!roiInRun.compare("roialt"))
					{
						landmarkInRun = true;
						std::cout << "You can not use roialt without landmark. Force to use lankmark." << endl;
					}
					else
					{
						landmarkInRun = false;
					}
				}
				else if(!tempStringInRun.compare("default"))
				{
					landmarkInRun = true;
				}
				else
				{
					landmarkInRun = true;
					std::cout << "Error landmark: put a default (yes).";
				}

				if (!facialMethodInRun.compare("cascade"))
				{
					std::cout << "Give me cascade (default, alt, alt2, lbp or lbp2)" << endl;
					std::cin.clear();
					std::cin.sync();
					std::getline(std::cin, tempStringInRun);
					if (!tempStringInRun.compare("default"))
					{
						cascadeChoseInRun = "default";
					}
					else if (!tempStringInRun.compare("alt"))
					{
						cascadeChoseInRun = "alt";
					}
					else if (!tempStringInRun.compare("alt2"))
					{
						cascadeChoseInRun = "alt2";
					}
					else if (!tempStringInRun.compare("lbp"))
					{
						cascadeChoseInRun = "lbp";
					}
					else if (!tempStringInRun.compare("lbp2"))
					{
						cascadeChoseInRun = "lbp2";
					}
					else
					{
						cascadeChoseInRun = "default";
						std::cout << "Errore cascade: put a default (default)." << endl;
					}
				}
				else
				{
					cascadeChoseInRun = "default";
				}

				std::cout << endl << "Facial chose: " << facialMethodInRun << endl;
				std::cout << "Hist chose: " << histTypeInRun << endl;
				std::cout << "ROI chose: " << roiInRun << endl;
				std::cout << "Landmark chose: " << landmarkInRun << endl;
				std::cout << "XML cascade chose: " << cascadeChoseInRun << endl << endl;

				if (sourceImage == 0) // static image
				{
					getFace(facialMethodInRun, histTypeInRun, 1, 0, roiInRun, landmarkInRun, cascadeChoseInRun);
				}
				else if (sourceImage == 1) // camera image
				{
					getFace(facialMethodInRun, histTypeInRun, 1, 1, roiInRun, landmarkInRun, cascadeChoseInRun);
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