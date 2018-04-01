#include "featureExtraction.h"

cv::Mat extractFeaturesFromSingleImage(std::string featuresExtractionAlgorithm)
{

	std::string inputFolder = baseDatabasePath + "/" + nameDataset + "/" + "temp";

	std::string inputFile = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + featuresExtractionAlgorithm + nameFileFeatures;
	FileStorage in(inputFile, FileStorage::READ);

	std::string dictionaryFile = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + nameDictionary;
	cv::FileStorage fsDictionary(dictionaryFile, cv::FileStorage::READ);

	std::string pcaFile = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + namePca;
	cv::FileStorage fsPca(pcaFile, cv::FileStorage::READ);

	cv::Mat image = imread(inputFolder + "/" + "imageTempROI.tiff", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat image = imread(inputFolder + "/" + "imageTempROI_alt.tiff", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat imageAlt = imread(inputFolder + "/" + "imageTempROI_alt.tiff", CV_LOAD_IMAGE_GRAYSCALE);

	cv::resize(image, image, Size(80, 80));
	//cv::resize(imageAlt, imageAlt, Size(80, 80));

	cv::Mat featuresVector;
	cv::Mat featuresExtracted = runExtractFeature(image, featuresExtractionAlgorithm);
	featuresVector.push_back(featuresExtracted);

	//cv::Mat featuresVectorAlt;
	//cv::Mat featuresExtractedAlt = runExtractFeature(imageAlt, featuresExtractionAlgorithm);
	//featuresVectorAlt.push_back(featuresExtractedAlt);

	int numberImages = 1;

	cv::Mat dictionary;
	fsDictionary["dictionary"] >> dictionary;

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
	cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);
	bowDE.setVocabulary(dictionary);

	cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();

	int dictionarySize = dictionary.rows;

	cv::Mat featuresDataOverBins = cv::Mat::zeros(numberImages, dictionarySize, CV_32FC1);

	std::vector<cv::KeyPoint> keypoints;
	detector->detect(image, keypoints);
	cv::Mat bowDescriptors;
	bowDE.compute(image, keypoints, bowDescriptors);

	bowDescriptors.copyTo(featuresDataOverBins);

	cv::PCA pca;

	pca.read(fsPca.root());

	int featureSize = pca.eigenvectors.rows;
	cv::Mat feature;
	for (int i = 0; i < numberImages; ++i) {
		feature = pca.project(featuresDataOverBins.row(i));
	}

	return feature;
}

void featureExtraction(std::string featuresExtractionAlgorithm)
{
	std::string inputFolder = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult;
	std::string outputFolder = baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult;

	// Prendo il file yml in cui sono riportati i percorsi delle immagini croppate.
	std::string inputFile = inputFolder + "/" + fileList;
	cv::FileStorage in(inputFile, cv::FileStorage::READ);

	//  Preparo il file di output in cui saranno inserite le features individuate.
	std::string outputFile = outputFolder + "/" + featuresExtractionAlgorithm + nameFileFeatures;
	cv::FileStorage ou(outputFile, cv::FileStorage::WRITE);

	std::string dictionaryFile = outputFolder + "/" + nameDictionary;
	cv::FileStorage fsDictionary(dictionaryFile, cv::FileStorage::WRITE);

	std::string pcaFile = outputFolder + "/" + namePca;
	FileStorage fsPca(pcaFile, FileStorage::WRITE);

	// Prendo il numero di immagini individuate da facialComponents.
	int numberImages = 0;
	in["number_of_image"] >> numberImages;

	// Dichiaro il vettore che conterrà le features.
	cv::Mat featuresVector;

	std::vector<cv::Mat> faceMatVector;

	long long startKFeautesExtraction = milliseconds_now();

	// Itero per tutte le immagini individuate e salvate in facialComponents
	for (int i = 0; i < numberImages; ++i)
	{
		// Scrivo in facePath i path alle immagini individuate in facialComponents.
		std::string facePath;
		in["image_" + std::to_string(i) + "_face"] >> facePath;

		// Carico l'immagine e la salvo in face.
		cv::Mat face = cv::imread(facePath, CV_LOAD_IMAGE_GRAYSCALE);
		resize(face, face, Size(80, 80));

		faceMatVector.push_back(face);

		cv::Mat featuresExtracted = runExtractFeature(face, featuresExtractionAlgorithm);

		// Inserisco nel vettore delle features le features che ho individuato con la funzione runExtractFeature. Spefico l'immagine e il nome dell'"estrattore".
		// Ogni elemento del vettore featuresVector contiene una matrice di dimensioni <keypoints>X128.
		// Le righe rappresentanto il numero di features estratte per quell'immagine, cioè i keypoints.
		// Le colonne sono un numero fisso, 128, dovuto all'implementazione con OpenCV (bin usati) e rappresentano i descriptors.
		featuresVector.push_back(featuresExtracted);
	}

	long long elapsedFeaturesExtraction = milliseconds_now() - startKFeautesExtraction;
	cout << "Time elapsed for extraction: " << elapsedFeaturesExtraction / 1000 << "s." << endl;

	int dictionarySize = 1000;
	cv::TermCriteria termCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1.0);
	int retries = 3;
	int centersFlags = KMEANS_PP_CENTERS;

	cv::BOWKMeansTrainer bowTrainer(dictionarySize, termCriteria, retries, centersFlags);

	bowTrainer.add(featuresVector);

	cv::Mat dictionary = bowTrainer.cluster();

	fsDictionary << "dictionary" << dictionary;

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
	cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);
	bowDE.setVocabulary(dictionary);

	cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();

	int numberTest = 0;
	cv::RNG random(cv::getTickCount());

	// Una matrice per accogliere le nuove features.
	// La grandezza deve essere rapportata al numero di immagini individuate e al numero di clusters scelto.
	cv::Mat featuresDataOverBins = cv::Mat::zeros(numberImages, dictionarySize, CV_32FC1);

	int counterLabelAngry = 0;
	int counterLabelDisgust = 0;
	int counterLabelFear = 0;
	int counterLabelHappy = 0;
	int counterLabelNeutral = 0;
	int counterLabelSad = 0;
	int counterLabelSurprised = 0;

	long long startBagOfWords = milliseconds_now();

	for (int i = 0; i < numberImages; ++i)
	{
		cv::Mat face = faceMatVector[i];

		std::vector<cv::KeyPoint> keypoints;
		detector->detect(face, keypoints);
		cv::Mat bowDescriptors;
		bowDE.compute(face, keypoints, bowDescriptors);

		// Prendo il path dell'attuale immagine che sto processando dal solito file.
		std::string path;
		in["image_" + std::to_string(i) + "_face"] >> path;

		// Prendo il nome del file.
		std::string filename = path.substr(inputFolder.length() + 1, path.length());

		// Si prende la parte di nostro interesse, cioè quella che specifica l'espressione dell'attuale immagine.
		string codeExpressionDataset = filename.substr(3, 2);

		int label = -1;
		if (!codeExpressionDataset.compare("AN")) { // Angry
			label = 0;
			++counterLabelAngry;
		}
		else if (!codeExpressionDataset.compare("DI")) { // Disgust
			label = 1;
			++counterLabelDisgust;
		}
		else if (!codeExpressionDataset.compare("FE")) { // Fear
			label = 2;
			++counterLabelFear;
		}
		else if (!codeExpressionDataset.compare("HA")) { // Happy
			label = 3;
			++counterLabelHappy;
		}
		else if (!codeExpressionDataset.compare("NE")) { // Neutral
			label = 4;
			++counterLabelNeutral;
		}
		else if (!codeExpressionDataset.compare("SA")) { // Sad
			label = 5;
			++counterLabelSad;
		}
		else if (!codeExpressionDataset.compare("SU")) { // Surprised
			label = 6;
			++counterLabelSurprised;
		}

		// Si scrive nel file di output le informazioni riguardandi la label.
		ou << "image_label_" + std::to_string(i) << label;

		// Si splitta il database in due parti, una per il test e una per il train.
		// Si utilizza una distribuzione uniforme per fare la scelta.
		double c = random.uniform(0., 1.);
		bool isTrain = true;
		if (c > 0.8) {
			isTrain = false;
			numberTest += 1;
		}

		// Si scrive nel file a quale dataset appartiene l'immagine corrente (train o test).
		ou << "image_is_train_" + to_string(i) << isTrain;

		bowDescriptors.copyTo(featuresDataOverBins.row(i));
	}

	long long elapsedBagOfWords = milliseconds_now() - startBagOfWords;
	cout << "Time elapsed for bag of words: " << elapsedBagOfWords / 1000 << "s." << endl;

	long long startPCA = milliseconds_now();

	// A questo punto si utilizza PCA per ridurre lo la dimensione dello spazio delle features.
	// Infatti, si era aumentata la dimensione per avere uniformità su tutte le immagini.
	// A questo punto, però, dobbiamo di nuovo ridurlo, perché il dataset non contiene poche immagine confrontate alla dimensione delle features.
	// Se non si facesse, il classificatore tenderebbe a fare overfitting sul training set e non riuscirebbe a generalizzare.
	// Si usa PCA, quindi, per ridurre lo spazio, tenere le features più importanti che hanno la maggiore varianza.
	cv::PCA pca(featuresDataOverBins, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.90);

	// Alcune dichiarazioni di variabili utili.
	int featureSize = pca.eigenvectors.rows;
	cv::Mat feature;
	// Itero su tutte le immagine del dataset.
	for (int i = 0; i < numberImages; ++i) {
		// Riduco la dimensione dello spazio delle features.
		feature = pca.project(featuresDataOverBins.row(i));
		// Si salva il risultato nel file di output, cioè una variabile mat che contiene le features dell'immagine i-esima.
		ou << "image_feature_" + to_string(i) << feature;
	}

	long long elapsedPCA = milliseconds_now() - startPCA;
	cout << "Time elapsed for PCA: " << elapsedPCA / 1000 << "s." << endl;

	// Il numero di features trovato.
	// E' il medesimo per ogni immagine.
	ou << "feature_size" << featureSize;
	// Il numero di immagini del dataset.
	ou << "number_of_image" << numberImages;
	// Il numero di label del dataset.
	ou << "number_of_label" << 7;
	ou << "label_0" << "Angry";
	ou << "label_1" << "Disgusted";
	ou << "label_2" << "Fear";
	ou << "label_3" << "Happy";
	ou << "label_4" << "Neutral";
	ou << "label_5" << "Sad";
	ou << "label_6" << "Surprised";
	// Il numero di immagini usate per il train.
	ou << "number_of_train" << numberImages - numberTest;
	// Il numero di immagini usate per il test.
	ou << "number_of_test" << numberTest;

	pca.write(fsPca);

	cout << "-------" << endl;
	cout << "Number angry: " << counterLabelAngry << " with label " << 0 << endl;
	cout << "Number disgust: " << counterLabelDisgust << " with label " << 1 << endl;
	cout << "Number Fear: " << counterLabelFear << " with label " << 2 << endl;
	cout << "Number Happy: " << counterLabelHappy << " with label " << 3 << endl;
	cout << "Number Neutral: " << counterLabelNeutral << " with label " << 4 << endl;
	cout << "Number Sad: " << counterLabelSad << " with label " << 5 << endl;
	cout << "Number Surprised: " << counterLabelSurprised << " with label " << 6 << endl;

	ou.release();
	in.release();
	fsDictionary.release();
	fsPca.release();
}

// Loader per gli "estrattori" di features.
cv::Mat runExtractFeature(cv::Mat image, std::string featureName) {
	cv::Mat descriptors;

	if (featureName.compare("kaze") == 0) {
		descriptors = extractFeaturesKaze(image);
	}
	else if (featureName.compare("sift") == 0) {
		descriptors = extractFeaturesSift(image);
	}
	else if (featureName.compare("surf") == 0) {
		descriptors = extractFeaturesSurf(image);
	}
	else if (featureName.compare("brisk") == 0) {
		descriptors = extractFeaturesBrisk(image);
	}
	else if (featureName.compare("daisy") == 0) {
		descriptors = extractFeaturesDaisy(image);
	}
	return descriptors;
}

cv::Mat extractFeaturesSift(cv::Mat image) {
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

	//cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> sift = cv::xfeatures2d::SIFT::create();
	cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> sift = cv::xfeatures2d::SIFT::create();
	sift->detect(image, keypoints, cv::Mat());
	sift->compute(image, keypoints, descriptors);

	return descriptors;
}

cv::Mat extractFeaturesSurf(Mat image) {
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

	//cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> surf = cv::xfeatures2d::SURF::create();
	cv::Ptr<cv::xfeatures2d::SurfDescriptorExtractor> surf = cv::xfeatures2d::SURF::create();
	surf->detect(image, keypoints, cv::Mat());
	surf->compute(image, keypoints, descriptors);

	return descriptors;
}

cv::Mat extractFeaturesKaze(cv::Mat image) {
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

	cv::Ptr<cv::DescriptorExtractor> kaze = cv::KAZE::create();
	kaze->detect(image, keypoints, cv::Mat());
	kaze->compute(image, keypoints, descriptors);

	return descriptors;
}

cv::Mat extractFeaturesBrisk(cv::Mat image) {
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

	cv::Ptr<cv::DescriptorExtractor> brisk = cv::BRISK::create();
	brisk->detect(image, keypoints, cv::Mat());
	brisk->compute(image, keypoints, descriptors);

	return descriptors;
}

cv::Mat extractFeaturesDaisy(Mat image) {
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

	cv::Ptr<cv::FeatureDetector> surf = cv::xfeatures2d::SURF::create();
	surf->detect(image, keypoints, cv::Mat());

	cv::Ptr<cv::DescriptorExtractor> daisy = cv::xfeatures2d::DAISY::create();
	daisy->compute(image, keypoints, descriptors);

	return descriptors;
}

cv::Mat extractFeaturesOrb(Mat image) {
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

	cv::Ptr<cv::Feature2D> fast = cv::ORB::create();
	fast->detect(image, keypoints, cv::Mat());
	fast->compute(image, keypoints, descriptors);

	return descriptors;
}