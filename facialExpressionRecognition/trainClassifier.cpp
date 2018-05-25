#include "trainClassifier.h"

// http://dlib.net/multiclass_classification_ex.cpp.html

float runClassifier(std::string algorithmName, std::string inputFeatures, std::string outputFolder)
{
	// Si apre il file in cui sono contenute le informazioni riguardo le features.
	FileStorage in(inputFeatures, FileStorage::READ);

	//Si dichiarano alcune variabili per inserire le informazioni prelevate dal file delle features.
	int numbersImage = 0;
	in["number_of_image"] >> numbersImage;
	int numbersTrain = 0;
	in["number_of_train"] >> numbersTrain;
	int numbersTest = 0;
	in["number_of_test"] >> numbersTest;
	int featureSize = 0;
	in["feature_size"] >> featureSize;
	int numbersLabel = 0;
	in["number_of_label"] >> numbersLabel;

	// Variabili per il dataset di train.
	// Un numero di righe pari al numero di immagini che fanno parte del dadaset di train.
	// Un numero di colonne pari al numero di features selezionate. E' un numero identico per ogni immagine, perché così abbiamo (e si doveva) fare.
	cv::Mat trainFeatures(numbersTrain, featureSize, CV_32FC1);
	// Un numero di righe pari al numero di immagini che fanno parte del dadaset di train.
	// Un numero di colonne pari a 1, perché ogni immagine ha una solo label possibile.
	cv::Mat trainLabels(numbersTrain, 1, CV_32SC1); // CV_32FC1 mia versione
	std::vector<string> trainPath;

	// Variabili per il dataset di test.
	// Medesimo discorso fatto per il dadaset di train.
	cv::Mat testFeatures(numbersTest, featureSize, CV_32FC1);
	cv::Mat testLabels(numbersTest, 1, CV_32SC1); // CV_32FC1 mia versione
	std::vector<string> testPath;

	// Due indici da utilizzare su i due dadaset perché l'indice i di iterazione del ciclo for non è sufficiente a gestire i due dadaset.
	int trainIndex = 0;
	int testIndex = 0;

	// Ancora dichiarazioni.
	cv::Mat feature;
	std::string path;
	int label;
	bool isTrain;

	// Itero su tutte le immagini che ho nel dadaset per separare le immagini.
	for (int i = 0; i < numbersImage; ++i)
	{
		// Estraggo le informazioni dell'immagine i-esima.
		in["image_feature_" + std::to_string(i)] >> feature;
		in["image_label_" + std::to_string(i)] >> label;
		in["image_path_" + std::to_string(i)] >> path;
		in["image_is_train_" + std::to_string(i)] >> isTrain;

		// Se l'immagine fa parte del dadaset di train.
		if (isTrain)
		{
			// Inserisco nella riga trainIndex di trainFeatures le features della i-esima immagine.
			feature.copyTo(trainFeatures.row(trainIndex));
			// Faccio la stessa cosa ma per la label.
			trainLabels.at<int>(trainIndex, 0) = label;
			// Stessa cosa, ma questa volta per il path.
			trainPath.push_back(path);
			// Incremento l'indice e passo alla prossima riga.
			trainIndex++;
		}
		else
		{
			// Faccio la stessa cosa, ma per dadaset di test.
			feature.copyTo(testFeatures.row(testIndex));
			testLabels.at<int>(testIndex, 0) = label;
			testPath.push_back(path);
			testIndex++;
		}
	}

	float accuracy;

	// Controllo che sia stato selezionato come algoritmo di classificazione svm e lancio la funzione.
	// I parametri sono abbastanza intutivi: ha bisogno di tutto.
	if (!algorithmName.compare("svm"))
	{
		accuracy = svmClassifier(numbersLabel, trainFeatures, trainLabels, testFeatures, testLabels);
	}
	else if (!algorithmName.compare("svmDlib"))
	{
		accuracy = svmClassifierDlib(numbersLabel, trainFeatures, trainLabels, testFeatures, testLabels);
	}
	else if (!algorithmName.compare("knn"))
	{
		int numberMaximumNeighbors = 2;
		accuracy = knnClassifier(numbersLabel, trainFeatures, trainLabels, testFeatures, testLabels, numberMaximumNeighbors);
	}
	else if (!algorithmName.compare("bayes"))
	{
		accuracy = bayesClassifier(numbersLabel, trainFeatures, trainLabels, testFeatures, testLabels);
	}
	else if (!algorithmName.compare("randomForest"))
	{
		accuracy = randomForestClassifier(numbersLabel, trainFeatures, trainLabels, testFeatures, testLabels);
	}
	else if (!algorithmName.compare("logisticRegression"))
	{
		accuracy = logisticRegressionClassifier(numbersLabel, trainFeatures, trainLabels, testFeatures, testLabels);
	}

	cout << "Accuracy = " << accuracy << endl;

	in.release();

	return accuracy;
}

float svmClassifierDlib(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels)
{
	/*typedef dlib::matrix<float, 7, 1> sample_type;
	typedef dlib::one_vs_one_trainer<dlib::any_trainer<sample_type> > ovo_trainer;
	ovo_trainer trainer;
	typedef dlib::polynomial_kernel<sample_type> poly_kernel;
	dlib::svm_nu_trainer<poly_kernel> poly_trainer;
	poly_trainer.set_kernel(poly_kernel(0.1, 1, 7));
	trainer.set_trainer(poly_trainer);

	std::vector<sample_type> samples;
	std::vector<sample_type> train;
	std::vector<sample_type> test;
	std::vector<float> labels;
	std::vector<float> labels1;
	std::vector<float> labels2;

	train.assign((sample_type*)trainFeatures.datastart, (sample_type*)trainFeatures.dataend);
	test.assign((sample_type*)testFeatures.datastart, (sample_type*)testFeatures.dataend);
	std::merge(train.begin(), train.end(), test.begin(), test.end(), samples.begin());

	labels1.assign((int*)trainLabels.datastart, (int*)trainLabels.dataend);
	labels2.assign((int*)testLabels.datastart, (int*)testLabels.dataend);
	std::merge(labels1.begin(), labels1.end(), labels2.begin(), labels2.end(), labels.begin());

	cout << "cross validation: \n" << dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
	dlib::one_vs_one_decision_function<ovo_trainer> df = trainer.train(trainFeatures, trainLabels);
	cout << "predicted label: " << df(samples[0]) << ", true label: " << labels[0] << endl;
	*/
	return 1.0;
}

float svmClassifier(int numberslabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels)
{

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	//svm->setKernel(cv::ml::SVM::RBF);
	svm->setKernel(cv::ml::SVM::LINEAR);
	Ptr<ml::TrainData> trainData = cv::ml::TrainData::create(trainFeatures, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);
	//svm->train(trainData);
	svm->trainAuto(trainData);

	svm->save(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + nameSVMModelTrained);

	// Array per contenere tutti i risultati delle label predette.
	// Una riga per ogni immagine che fa parte del dataset di test.
	// Una colonna per ogni label differente, nel nostro caso 7 (7 espressioni facciali).
	cv::Mat predicted = cv::Mat::zeros(testLabels.rows, numberslabel, CV_32FC1); // CV_32SC1
	for (int i = 0; i < testFeatures.rows; i++) {
		// Si prende la i-esima feature dell'i-esima immagine.
		cv::Mat sample = testFeatures.row(i);
		// Si prende la label predetta per l'i-esima features.
		float predict = svm->predict(sample);
		// Si mette a 1 l'elemento dell'array dell'i-esima riga e della colonna della label predetta.
		predicted.at<float>(i, (int)predict) = 1.0f; //predicted.at<int>(i, (int)predict) = 1.0f;
	}

	float accuracy = computeAccuracy(predicted, testLabels);

	return accuracy;
}

float knnClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels, int k)
{
	cv::Ptr<ml::KNearest> knn = ml::KNearest::create();
	cv::Ptr<ml::TrainData> trainData = cv::ml::TrainData::create(trainFeatures, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);
	knn->setDefaultK(k);
	knn->setIsClassifier(true);
	knn->train(trainData);

	knn->save(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + nameKnnModelTrained);

	cv::Mat predicted = cv::Mat::zeros(testLabels.rows, numbersLabel, CV_32FC1); // CV_32SC1
	for (int i = 0; i < testFeatures.rows; i++)
	{
		cv::Mat sample = testFeatures.row(i);
		cv::Mat predict;
		knn->findNearest(sample, k, predict);
		predicted.at<float>(i, predict.at<float>(0, 0)) = 1.0f; // predicted.at<int>(i, predict.at<float>(0, 0)) = 1.0f;

		/*
		float predict = knn->predict(sample);
		predicted.at<int>(i, (int)predict) = 1.0f;
		*/
	}

	float accuracy = computeAccuracy(predicted, testLabels);

	return accuracy;
}

float bayesClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels) 
{
	cv::Ptr<cv::ml::NormalBayesClassifier> bayes = cv::ml::NormalBayesClassifier::create();
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainFeatures, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);
	bayes->train(trainData);

	bayes->save(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + nameBayesModelTrained);

	cv::Mat predicted = cv::Mat::zeros(testLabels.rows, numbersLabel, CV_32FC1); // CV_32SC1
	for (int i = 0; i < testFeatures.rows; i++)
	{
		cv::Mat sample = testFeatures.row(i);
		cv::Mat output, outputProb;
		bayes->predictProb(sample, output, outputProb);

		predicted.at<float>(i, output.at<int>(0, 0)) = 1.0f; // predicted.at<float>(i, output.at<unsigned int>(0, 0)) = 1.0f;

		/*
		float predict = bayes->predict(sample);
		predicted.at<int>(i, (int)predict) = 1.0f;
		*/
	}

	float accuracy = computeAccuracy(predicted, testLabels);

	return accuracy;
}

float randomForestClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels)
{
	cv::Ptr<cv::ml::RTrees> randomForest = cv::ml::RTrees::create();
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainFeatures, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);
	randomForest->train(trainData);

	randomForest->save(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + nameRandomForestModelTrained);

	cv::Mat predicted = cv::Mat::zeros(testLabels.rows, numbersLabel, CV_32FC1); // CV_32SC1
	for (int i = 0; i < testFeatures.rows; i++) {
		cv::Mat sample = testFeatures.row(i);
		float predict = randomForest->predict(sample);
		predicted.at<float>(i, (int)predict) = 1.0f; // predicted.at<int>(i, (int)predict) = 1.0f;
	}

	float accuracy = computeAccuracy(predicted, testLabels);

	return accuracy;
}

float logisticRegressionClassifier(int numbersLabel, cv::Mat trainFeatures, cv::Mat trainLabels, cv::Mat testFeatures, cv::Mat testLabels)
{
	cv::Ptr<cv::ml::LogisticRegression> logisticRegression = cv::ml::LogisticRegression::create();
	//cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainFeatures, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);

	logisticRegression->setLearningRate(0.001);
	logisticRegression->setIterations(10);
	logisticRegression->setRegularization(cv::ml::LogisticRegression::REG_L2);
	logisticRegression->setTrainMethod(cv::ml::LogisticRegression::BATCH);
	logisticRegression->setMiniBatchSize(1);

	trainFeatures.convertTo(trainFeatures, CV_32F);
	trainLabels.convertTo(trainLabels, CV_32F);

	logisticRegression->train(trainFeatures, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);
	//logisticRegression->train(trainData);

	logisticRegression->save(baseDatabasePath + "/" + nameDataset + "/" + nameDirectoryResult + "/" + nameLogisticRegressionModelTrained);

	cv::Mat predicted = cv::Mat::zeros(testLabels.rows, numbersLabel, CV_32FC1); // CV_32FC1
	for (int i = 0; i < testFeatures.rows; i++) {
		cv::Mat sample = testFeatures.row(i);
		float predict = logisticRegression->predict(sample);
		predicted.at<float>(i, (int)predict) = 1.0f; // predicted.at<int>(i, (int)predict) = 1.0f;
	}

	float accuracy = computeAccuracy(predicted, testLabels);

	return accuracy;
}

float computeAccuracy(cv::Mat predicted, cv::Mat actual) {
	// Si controlla che il numero di righe delle due matrici sia identico.
	assert(predicted.rows == actual.rows);
	// Counter per le label corrette.
	int trueLabel = 0;
	// Counter per la label errate.
	int falseLabel = 0;

	// Counter che quantifica il numero di immagini classificate correttamente con Angry.
	int trueAngerLabel = 0;
	// Counter che quantifica il numero di immagini che sono state classificate come Angry ma che non lo sono.
	int falsePositiveAngerLabel = 0;
	// Un vettore che quantifica il numero di immagini che sono state classificate come Angry ma che non lo erano, ma erano.
	cv::Mat angerInsteadLabel = cv::Mat::zeros(predicted.rows, predicted.cols, CV_32S);

	int trueDisgustLabel = 0;
	int falsePositiveDisgustLabel = 0;
	cv::Mat disgustInsteadLabel = cv::Mat::zeros(predicted.rows, predicted.cols, CV_32S);

	int trueFearLabel = 0;
	int falsePositiveFearLabel = 0;
	cv::Mat fearInsteadLabel = cv::Mat::zeros(predicted.rows, predicted.cols, CV_32S);

	int trueHappyLabel = 0;
	int falsePositiveHappyLabel = 0;
	cv::Mat happyInsteadLabel = cv::Mat::zeros(predicted.rows, predicted.cols, CV_32S);

	int trueSadLabel = 0;
	int falsePositiveSadLabel = 0;
	cv::Mat sadInsteadLabel = cv::Mat::zeros(predicted.rows, predicted.cols, CV_32S);

	int trueNeutralLabel = 0;
	int falsePositiveNeutralLabel = 0;
	cv::Mat neutralInsteadLabel = cv::Mat::zeros(predicted.rows, predicted.cols, CV_32S);

	int trueSurpriseLabel = 0;
	int falsePositiveSurpriseLabel = 0;
	cv::Mat surpriseInsteadLabel = cv::Mat::zeros(predicted.rows, predicted.cols, CV_32S);

	// Itero sulle label reali (sulle righe).
	for (int i = 0; i < actual.rows; ++i) 
	{
		float max = -1000000000000.0f;
		int colonIndex = -1;

		// Itero sulle label predette (sulle colonne).
		for (int j = 0; j < predicted.cols; ++j)
		{
			// Il valore della label predetta alla riga i-esima della j-esima colonna.
			float value = predicted.at<float>(i, j); // float value = predicted.at<int>(i, j);
			// Se il valore è maggiore.
			if (value > max)
			{
				// Il nuovo massimo.
				max = value;
				// Indice colonna del nuovo massimo.
				colonIndex = j;
			}
		}

		// Al termine del ciclo for innestato.
		// colonIndex contiene l'indice dell'unico elemento a 1 della riga i-esima.
		// max contiene il valore contenuto nella matrice predicted alla riga i-esima e j-esima colonna.

		// Il valore della label reale.
		int truth = (int)actual.at<int>(i, 0);
		// Se il valore reale è uguale a quello predetto.
		// Si prende l'indice (cls) e non max, perché il valore 1 contenuto nella matrice predicted è solo un segnaposto per la colonna della label predetta.
		if (colonIndex == truth)
		{
			trueLabel++;

			switch (colonIndex)
			{
			case 0:
				trueAngerLabel++;
				break;
			case 1:
				trueDisgustLabel++;
				break;
			case 2:
				trueFearLabel++;
				break;
			case 3:
				trueHappyLabel++;
				break;
			case 4:
				trueNeutralLabel++;
				break;
			case 5:
				trueSadLabel++;
				break;
			case 6:
				trueSurpriseLabel++;
				break;
			default:
				return -1;
			}
		}
		else
		{
			falseLabel++;

			switch (colonIndex)
			{
			case 0:
				falsePositiveAngerLabel++;
				angerInsteadLabel.at<int>(i, truth)++;
				break;
			case 1:
				falsePositiveDisgustLabel++;
				disgustInsteadLabel.at<int>(i, truth)++;
				break;
			case 2:
				falsePositiveFearLabel++;
				fearInsteadLabel.at<int>(i, truth)++;
				break;
			case 3:
				falsePositiveHappyLabel++;
				happyInsteadLabel.at<int>(i, truth)++;
				break;
			case 4:
				falsePositiveNeutralLabel++;
				neutralInsteadLabel.at<int>(i, truth)++;
				break;
			case 5:
				falsePositiveSadLabel++;
				sadInsteadLabel.at<int>(i, truth)++;
				break;
			case 6:
				falsePositiveSurpriseLabel++;
				surpriseInsteadLabel.at<int>(i, truth)++;
				break;
			default:
				return -1;
			}
		}
	}

	std::cout << "-------" << endl;
	std::cout << "Naive matrix confusion" << endl << endl;
	std::cout << "Angry classification: " << endl;
	std::cout << "\t Hit: " << trueAngerLabel << endl;
	std::cout << "\t False positive: " << falsePositiveAngerLabel << endl << endl;
	std::cout << "Disgusted classification: " << endl;
	std::cout << "\t Hit: " << trueDisgustLabel << endl;
	std::cout << "\t False positive: " << falsePositiveDisgustLabel << endl << endl;
	std::cout << "Fear classification: " << endl;
	std::cout << "\t Hit: " << trueFearLabel << endl;
	std::cout << "\t False positive: " << falsePositiveFearLabel << endl << endl;
	std::cout << "Happy classification: " << endl;
	std::cout << "\t Hit: " << trueHappyLabel << endl;
	std::cout << "\t False positive: " << falsePositiveHappyLabel << endl << endl;
	std::cout << "Neutral classification: " << endl;
	std::cout << "\t Hit: " << trueNeutralLabel << endl;
	std::cout << "\t False positive: " << falsePositiveNeutralLabel << endl << endl;
	std::cout << "Sad classification: " << endl;
	std::cout << "\t Hit: " << trueSadLabel << endl;
	std::cout << "\t False positive: " << falsePositiveSadLabel << endl << endl;
	std::cout << "Surprised classification: " << endl;
	std::cout << "\t Hit: " << trueSurpriseLabel << endl;
	std::cout << "\t False positive: " << falsePositiveSurpriseLabel << endl << endl;

	for (int i = 0; i < predicted.cols; ++i)
	{
		for (int j = 0; j < predicted.rows; ++j)
		{
			for (int k = 0; k < predicted.cols; ++k)
			{
				bool modified = false;
				switch (i)
				{
				case 0:
					if ((int)angerInsteadLabel.at<int>(j, k) != 0)
					{
						std::cout << "Supposed angry but was " << endl;
						modified = true;
					}
					break;
				case 1:
					if (disgustInsteadLabel.at<int>(j, k) != 0)
					{
						std::cout << "Supposed disgust but was " << endl;
						modified = true;
					}
					break;
				case 2:
					if (fearInsteadLabel.at<int>(j, k) != 0)
					{
						std::cout << "Supposed fear but was " << endl;
						modified = true;
					}
					break;
				case 3:
					if (happyInsteadLabel.at<int>(j, k) != 0)
					{
						std::cout << "Supposed happy but was " << endl;
						modified = true;
					}
					break;
				case 4:
					if (neutralInsteadLabel.at<int>(j, k) != 0)
					{
						std::cout << "Supposed neutral but was " << endl;
						modified = true;
					}
					break;
				case 5:
					if (sadInsteadLabel.at<int>(j, k) != 0)
					{
						std::cout << "Supposed sad but was " << endl;
						modified = true;
					}
					break;
				case 6:
					if (surpriseInsteadLabel.at<int>(j, k) != 0)
					{
						std::cout << "Supposed surprise but was " << endl;
						modified = true;
					}
					break;
				default:
					return -1;
				}
				if (modified)
				{
					switch (k)
					{
					case 0:
						std::cout << "\tangry" << endl;
						break;
					case 1:
						std::cout << "\tdisgust" << endl;
						break;
					case 2:
						std::cout << "\tfear" << endl;
						break;
					case 3:
						std::cout << "\thappy" << endl;
						break;
					case 4:
						std::cout << "\tneutral" << endl;
						break;
					case 5:
						std::cout << "\tsad" << endl;
						break;
					case 6:
						std::cout << "\tsurprised" << endl;
						break;
					default:
						return -1;
					}
				}
			}
		}
	}

	float precision = 0; // true positive / (true positive + false positive)
	float recall = 0; // true positive / (true positive + false negative)
	float f1Score = 0; // 2 * (precision * recall) / (precision + recall)

	std::cout << "Fin." << endl;
	std::cout << "-------" << endl;

	return (float)((trueLabel * 1.0) / (trueLabel + falseLabel) * 100);
}