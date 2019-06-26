#pragma comment(lib, "winmm.lib")

#include <Windows.h>
#include "kmeans.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cctype>
#include <stdlib.h>
#include <limits>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>



using namespace std;

void Run(int tC,  const int N, const int D, const int I, int& iteration, const double T, const int R, string method, string internal, vector <vector <double> >file, fstream& outputFile) {
	vector<double> RunSSE;
	/*outputFile << "K = " << K << endl;
	cout << "K = " << K << endl;*/
	/*cout << "K = " << K << endl;
	outputFile << "K = " << K << endl;*/
	for (int r = 1; r <= R; r++) {
		//Initialize the object for the clusters
		KMeans clusters = KMeans(tC, N, D, I, T, file);
		//Initialize/Reset the counter for the current iteration
		iteration = 0;
		//Initialize the SSE values of the clusters to zero
		for (int i = 0; i < I; i++) {
			clusters.SSE[i] = 0;
		}

		//outputFile << "Run " << r << ": " << "\n";
		cout << "Processing Run " << r << ": " << "\n";

		//Cluster initialization methods from Phase Two. Only maximin is used for Phase 3. 
		/*if (method == "part") {
			clusters.randomPartition();
		}
		else if (method == "maximin") {
			clusters.maxiMin();
		}*/

		//Assign the data points randomly
		clusters.updatePoints();
		//Calculate the initial centroids
		clusters.calculateCentroids(iteration);

		//Loop: Update point assignments and centroid calculations until convergence.
		do {
			iteration++;
			clusters.updatePoints();
			clusters.calculateCentroids(iteration);
			//outputFile << "Iteration " << iteration << ": SSE= " << clusters.SSE[iteration] << endl;


		} while ((iteration <= I) &&
			((clusters.SSE[iteration - 1] - clusters.SSE[iteration]) / (clusters.SSE[iteration - 1])) > T);

		//Push the converged SSE for the current run to a vector of Run SSEs
		/*clusters.SSE.erase(remove(clusters.SSE.begin(), clusters.SSE.end(), 0), clusters.SSE.end());
		RunSSE.push_back(clusters.SSE.back());*/

		/*if (internal == "SW") {
			clusters.calculateSW();
			if (r == R) {
				cout << "SW(" << K << ") = " << clusters.SW << endl;
				outputFile << "SW(" << K << ") = " << clusters.SW << endl;
			}
		}
		else if (internal == "CH") {
			clusters.calculateCH(iteration);
			if (r == R) {
				cout << "CH(" << K << ") = " << clusters.CH << endl;
				outputFile << "CH(" << K << ") = " << clusters.CH << endl;
			}
		}
		else if (internal == "DB") {
			clusters.DB = clusters.calculateDB();
			if (r == R) {
				
				cout << "DB(" << K << ") = " << clusters.DB << endl;
				outputFile << "DB(" << K << ") = " << clusters.DB << endl;
			}
		}*/

		//For Phase Four, implement external validity measures
		clusters.getExternalMeasures();
		clusters.Rand();
		clusters.Jaccard();
		clusters.FM();

		//Output results to file
		cout << "Rand: " << clusters.exRand << " Jaccard: " << clusters.exJaccard << " Fowlkes-Mallows: " << clusters.exFM << endl;
		outputFile  << "Rand: " << clusters.exRand <<  " Jaccard: "  << clusters.exJaccard << " Fowlkes-Mallows: "<< clusters.exFM << endl;

	}

	//Reset buffer to standard output
	//cout.rdbuf(coutbuf);
	cout << "Done." << endl;


	////Get the index and element of the best SSE
	//auto bestSSE = *min_element(RunSSE.begin(), RunSSE.end());
	//auto bestIndex = min_element(RunSSE.begin(), RunSSE.end()) - RunSSE.begin();

	/*cout << "Best Run: " << bestIndex + 1 << " SSE= " << bestSSE << " " << endl;
	outputFile << "\nBest Run: " << bestIndex + 1 << " SSE= " << bestSSE << " " << endl;*/
}


//Our readFile function has been modified for Phase Four to account for the true cluster/s
//The structure of the input datasets have been modified from previous phases
vector <vector <double> > readFile(string F, int& N, int& D,int&tC) {
	fstream inputFile;
	vector <vector <double>> file;
	//convert filename to string
	inputFile.open(F.c_str());
	inputFile >> N;
	inputFile >> D;
	inputFile >> tC;
	file.resize(N, vector<double>(D, 0));
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < D; j++) {
			inputFile >> file[i][j];
		}
	}
	inputFile.close();
	return file;
}
//Implementation of Min-Max data normalization
vector <vector <double>> normalizeMinMax(vector <vector <double> > file, int& N, int& D) {
	//Initialize vectors
	vector <double> min, max;
	vector <vector <double> > normMinMax;
	normMinMax.resize(N, vector<double>(D, 0));
	min.resize(D, 0);
	max.resize(D, 0);

	//Initialize the current max and min
	for (int i = 0; i < D; i++) {
		min[i] = file[0][i];
		max[i] = file[0][i];
	}

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < D; j++) {
			//Detect new min/max
			if (file[i][j] < min[j])
				min[j] = file[i][j];
			if (file[i][j] > max[j])
				max[j] = file[i][j];
		}
	}

	//Normalize the dataset based on max and minimum
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			//norm formula
			normMinMax[i][j] = ((file[i][j] - min[j]) / (max[j] - min[j]));
		}
	}

	//Return the normalized data
	return normMinMax;

}
//Implementation of z-score data normalization
vector <vector <double>> normalizeZScore(vector <vector <double> > file, int& N, int& D) {

	//Initialize values and vectors
	vector <double> sum, mean, stDev, alpha;
	vector <vector <double> > normZScore;
	normZScore.resize(N, vector<double>(D, 0));
	sum.resize(D, 0);
	mean.resize(D, 0);
	stDev.resize(D, 0);
	alpha.resize(D, 0);

	//Get the sum of the dataset
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			sum[j] += file[i][j];
		}
	}

	//Get the mean of the dataset
	for (int i = 0; i < D; i++) {
		mean[i] = sum[i] / N;
	}
	//Get the alpha for the dataset
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < D; j++) {
			alpha[j] += (file[i][j] - mean[j]) * (file[i][j] - mean[j]);
		}
	}
	//Get the standard deviation for the dataset
	for (int i = 0; i < D; i++) {
		stDev[i] = sqrt(alpha[i] / (N - 1.0));
	}
	//Finally, get the z-score for the normalized dataset
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < D; j++) {
			//formula for z-score
			normZScore[i][j] = (file[i][j] - mean[j]) / stDev[j];
		}
	}
	//Return the z-score normalized dataset
	return normZScore;

}



int main(int argc, const char* argv[]) {

	//Kmeans Clustering Phase Three:
	//the first major difference is that K, the number of clusters, is not user-defined.

	/****USAGE CHECKS **********/

	cout << "Hello." << endl;
	int N = 0;
	int D = 0;
	int iteration = 0;
	int trueClusters = 0;
	//Check for correct number of args when running from command-line
	if (argc < 5) {
		cerr << "ERROR: Invalid number of arguments." << endl
			<< "Usage: (DataClusteringPhase1.exe) test <file (F)> "
			<< "<Maximum number of iterations (I)> <Convergence threshold (T)> <Number of Runs (R)> ";
		exit(EXIT_FAILURE); //Terminate Program
	}
	//Open data file for reading
	fstream F(argv[2], ios::in);
	if (!F) {
		cerr << "ERROR: File could not be opened or file does not exist.";
		exit(EXIT_FAILURE); //Terminate Program
	}
	//const auto K = atoi(argv[3]);
	////Check for valid number of clusters
	//if (K <= 1) {
	//	cerr << "ERROR: Invalid input for number of clusters. Must be a positive integer greater than one";
	//	exit(EXIT_FAILURE);
	//}
	const auto I = atoi(argv[3]);
	//Check for valid number of iterations
	if (I <= 10) {
		cerr << "ERROR: Invalid input for number of iterations. Must be a positive integer greater than one." << endl
			<< "This program allows for a minimum of ten iterations per run.";
		exit(EXIT_FAILURE);
	}
	const double T = atof(argv[4]);
	//Check for valid convergence threshold
	if (T < 0) {
		cerr << "ERROR: Invalid convergence threshold. Must be a non-negative real number." << endl;
		exit(EXIT_FAILURE);
	}
	const auto R = atoi(argv[5]);
	//Check for valid number of runs
	if (R < 1) {
		cerr << "ERROR: Invalid input for number of runs. Must be a positive integer.";
		exit(EXIT_FAILURE);
	}

	cout << "All inputs are valid. Beginning kmeans algorithm." << endl;

	//Use a 2D vector to read the input data files
	vector <vector <double> > file, fileNormMinMax, fileNormZscore;




	//Use the simple random seed RNG
	srand((unsigned int)time(NULL));






	/****USAGE CHECKS **********/
	string temp = argv[2];
	file = readFile(temp, N, D,trueClusters);
	//The range of K values is now defined at file read.
	//The maximum number of clusters is simply the closest integer to sqrt(N/2).
	/*double minK = 2;
	double maxK = round(sqrt(N / 2));
	double K = minK;*/
	//For Phase Four, MinMax normalization is applied to the dataset before clustering
	fileNormMinMax = normalizeMinMax(file, N, D);
	//fileNormZscore = normalizeZScore(file, N, D);

	cout << "Outputting to File..." << endl;
	string outputfileName = "Output_" + temp;
	fstream outputFile(outputfileName, ios::out);
	if (!outputFile) {
		cerr << "ERROR: output file was not created.";
		exit(EXIT_FAILURE); //Terminate Program
	}


	Run(trueClusters, N, D, I, iteration, T, R, "", "", fileNormMinMax, outputFile);
	//outputFile.close();
	//streambuf *coutbuf = cout.rdbuf();
	//cout.rdbuf(outputFile.rdbuf()); //Redirect cout to output text file

	//Display input parameters at beginning  of output file
	/*cout << "Min-Max Normalization/Maximin Initialization" << endl;
	outputFile << "Min-Max Normalization/Maximin Initialization" << endl;
	outputFile << temp << " " << "Minimum # Clusters: " << minK << " Maximum # clusters: " << maxK << " " << I << " " << T << " " << R << endl << endl;
	cout << temp << " " << "Minimum # Clusters: " << minK << " Maximum # clusters: " << maxK << " " << I << " " << T << " " << R << endl << endl;*/


	/*cout << "SW internal validity: " << endl;
	outputFile <<"SW internal validity: " << endl;
	while (K <= maxK) {

		Run(K, minK, maxK, N, D, I, iteration, T, R, "maximin", "SW", fileNormMinMax, outputFile);
		K++;
	}
	K = minK;
	cout << "CH internal validity: " << endl;
	outputFile << "CH internal validity: " << endl;
	while (K <= maxK) {

		Run(K, minK, maxK, N, D, I, iteration, T, R, "maximin", "CH", fileNormMinMax, outputFile);
		K++;
	}
	K = minK;
	cout << "DB internal validity: " << endl;
	outputFile << "DB internal validity: " << endl;
	while (K <= maxK) {

		Run(K, minK, maxK, N, D, I, iteration, T, R, "maximin", "DB", fileNormMinMax, outputFile);
		K++;
	}*/


	//Play sound after program is finished. Useful for running the program in the background.
	PlaySound(TEXT("coins.wav"), NULL, SND_FILENAME);






	return 0;
}


