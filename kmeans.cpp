#include "kmeans.h"

#include <fstream>
#include <string>
#include <vector>


using namespace std;


//Constructor: Get input and file parameters and initialize vectors for Kmeans algorithm operations

KMeans::KMeans(int numClusters, int numPoints, int Dim, int Iterations, double Threshold, vector < vector<double>> data) {
	//For Phase Four, the constructor is modified to account for the true cluster/s in the target dataset.

	K = numClusters;
	tC = numClusters; //initialize true number of clusters to input K
	N = numPoints;
	D = (Dim - 1);
	I = Iterations;
	T = Threshold;
	//inputFile = data;
	maxSE = 0;
	SW = 0, CH = 0, DB = 0;
	//Phase one uses the Random Selection method for point assignments
	int randSelection;

	//Resize inputFile vector to match size of input dataset
	inputFile.resize(N, vector<double>(D, 0));
	//Resize vector for number of true clusters
	trueK.resize(N, 0);
	//initialize true cluster values
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			inputFile[i][j] = data[i][j];
		}
		trueK[i] = data[i][D];
	}
	//Rand external Validation
	int randIndex = 0;

	//Initialize vectors
	centroids.resize(K, vector<double>(D, 0)); //Initialize K centroids with D dimension
	//Intitialize SE values
	SE.resize(K);
	SSE.resize(I);
	maxSSE.resize(D);
	//Initialize the cluster sum
	sumK.resize(N);
	dataPoints.resize(N, vector <double>(K, 0));


	//Random Selection Method from Phase One is reused for Phase Four
	for (int i = 0; i < K; i++) {
		randSelection = rand() % N;
		centroids[i] = inputFile[randSelection];
	}


}
//Implementation of Random Partition method
//Simply assign each point to a random cluster
void KMeans::randomPartition() {

	int partition, total;
	double sum;

	dataPoints.resize(N, vector<double>(K, 0));
	for (int i = 0; i < N; i++) {
		//Pick a random partition for each point
		partition = rand() % K;
		dataPoints[i][partition] = 1;
	}

	for (int i = 0; i < K; i++) {

		for (int j = 0; j < D; j++) {
			sum = 0;
			total = 0;
			for (int l = 0; l < N; l++) {
				if (dataPoints[l][i] == 1) {
					sum = sum + inputFile[l][j];
					total++;
				}
			}
			//Assign centroids
			centroids[i][j] = sum / total;
		}

	}
}
//Implementation of the Maximin method.
//Choose an arbitrary first center for the first iteration.
//for the next  iteration, find the greatest Euclidean distance from the previously selected centers.
//Finally, find the greatest min from the first and second centers.
//Center i is the point x with the greatest min (d(x,c1),d(x,c2)...,d(x,c[i-1])) value.
void KMeans::maxiMin() {
	int randSelection;
	int point = 1;
	int m;
	double diff, max;

	//Initialize vectors
	vector <double> minima;
	vector <vector <double> > dist;

	dist.resize(N, vector <double>(K, 0));
	randSelection = rand() % N;
	//Pick the first centroid arbitrarily
	centroids[0] = inputFile[randSelection];
	//Get the Euclidean Distance
	do {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < point; j++) {
				diff = 0;
				for (int l = 0; l < D; l++) {
					diff += ((inputFile[i][l] - centroids[j][l]) * (inputFile[i][l] - centroids[j][l]));
				}
				dist[i][j] = sqrt(diff);
			}
		}
		//Get minimum distance
		minima.resize(N, 0);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < point; j++) {
				if (dist[i][j] < minima[i]) {
					minima[i] = dist[i][j];
				}
			}
		}
		m = 0;
		//Get maximum distance
		max = minima[0];
		for (int i = 0; i < N; i++) {
			if (minima[i] > max) {
				max = minima[i];
				m = i;
			}
		}
		//Get Next centroid
		centroids[point] = inputFile[max];
		point++;

	} while (point < K);

}

//Inititialize/Update point assignments using Euclidean Distance
void KMeans::updatePoints() {

	//Euclidean distance is used to update point assignments
	vector <vector <double> > dist;
	//Store the current difference and the minimum distance
	double diff;
	double min;

	//Resize distance vector and initialize/reset values
	dist.resize(N, vector <double>(K, 0));
	for (int i = 0; i < N; i++)
	{
		dataPoints[i][0] = 1;
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 1; j < K; j++)
		{
			dataPoints[i][j] = 0;
		}
	}

	//Get the Euclidean Distance
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K; j++)
		{
			diff = 0;
			for (int l = 0; l < D; l++)
			{
				diff += ((inputFile[i][l] - centroids[j][l]) * (inputFile[i][l] - centroids[j][l]));
			}
			dist[i][j] = sqrt(diff);
		}
	}


	//Get minimum distance
	for (int i = 0; i < N; i++)
	{
		min = dist[i][0];
		for (int j = 1; j < K; j++)
		{
			if (dist[i][j] < min)
			{
				min = dist[i][j];
				for (int l = 0; l < K; l++)
				{
					dataPoints[i][l] = 0;
				}
				dataPoints[i][j] = 1;
			}
		}
	}

	//Get the cluster sums
	for (int i = 0; i < K; i++)
	{
		sumK[i] = 0;
		for (int j = 0; j < N; j++)
		{
			sumK[i] += dataPoints[j][i];
		}
	}



}

//Determine the centroids for each cluster.
//Also get the performance metric (SSE) for the current iteration
void KMeans::calculateCentroids(int iteration) {

	double sum, diff;
	int total;

	//Determine centroids of each cluster
	for (int i = 0; i < K; i++) {

		for (int j = 0; j < D; j++) {
			//Intitialize/reset sum and total
			sum = 0;
			total = 0;
			for (int l = 0; l < N; l++) {
				if (dataPoints[l][i] == 1) {
					sum = sum + inputFile[l][j];
					total++;
				}
			}
			//Generate centroids
			centroids[i][j] = sum / total;
		}
		if (sumK[i] == 0) {
			centroids[i] = maxSSE;
		}

	}


	maxSE = 0;
	maxSSE = inputFile[0];

	//Get the SSE for each iteration
	for (int i = 0; i < K; i++) {

		diff = 0;
		SE[i] = 0;

		for (int j = 0; j < N; j++) {
			sum = 0;
			total = 0;
			for (int l = 0; l < D; l++) {
				if (dataPoints[j][i] == 1) {
					diff += ((inputFile[j][l] - centroids[i][l]) * (inputFile[j][l] - centroids[i][l]));
				}
				if (diff > maxSE) {
					maxSSE = inputFile[i];
					maxSE = diff;
				}
			}
		}
		SE[i] = diff;
		SSE[iteration] += SE[i];

	}

}

//Phase Three: Internal Validity Measures
//The equations for these measures are defined around pg. 440-450
//of Data Mining and Analysis: Fundamental Concepts and Algorithms
//Mohammed J.Zaki and Wagner Meira, Jr.
//http://www.dataminingbook.info/pmwiki.php/Main/BookDownload



//Calculates the Silhouette Width (SW) internal validity index
void KMeans::calculateSW() {

	//Initialization
	//The terms of the equation can be stored as vectors
	vector <double> muMinOut, muMax;
	//local variables for the difference, near, and maximum.
	double diff, near, max;
	//iterators for number of points within and outside a given cluster.
	int inside = 0, outside = 0;

	//Dynamically resize vectors based on the number of tuples in the target dataset.
	muMinOut.resize(N);
	muMax.resize(N);
	nearest.resize(N, vector<double>(K, 0));
	//Initialize SW index
	SW = 0;

	//Iterate through every point in the dataset
	for (int i = 0; i < N; i++) {
		//Check for cluster assignments
		if (dataPoints[i][0] == 0) {
			nearest[i][0] = 1;
			diff = 0;
			//Accumulate the difference
			for (int j = 0; j < D; j++) {
				diff += (inputFile[i][j] - centroids[0][j]) * (inputFile[i][j] - centroids[0][j]);
			}
			near = diff;
		}
		else {
			nearest[i][1] = 1;
			diff = 0;
			for (int j = 0; j < D; j++) {
				//Accumulate the difference
				diff += (inputFile[i][j] - centroids[1][j]) * (inputFile[i][j] - centroids[1][j]);
			}
			near = diff;
		}
		//Check K # of clusters
		for (int l = 0; l < K; l++) {
			diff = 0;
			if (dataPoints[i][l] != 1) {
				//Accumulate the difference
				for (int j = 0; j < D; j++) {
					diff += (inputFile[i][j] - centroids[l][j]) * (inputFile[i][j] - centroids[l][j]);
				}
				if (diff < near) {
					near = diff;
					for (int m = 0; m < K; m++) {
						nearest[i][m] = 0;
					}
					nearest[i][l] = 1;
				}
			}
		}
	}

	//Iteratively determine the number of points outside and calculate the term muMinOut
	for (int i = 0; i < N; i++) {
		diff = 0;
		for (int l = 0; l < N; l++) {
			if (l != i && dataPoints[l] == nearest[i]) {
				for (int j = 0; j < D; j++) {
					diff += abs(inputFile[i][j] - inputFile[l][j]);
					outside++;
				}
			}
		}
		//Calculate the term muMinOut for data point[i]
		muMinOut[i] = diff / outside;

	}
	//Iteratively determine the number of points inside and calculate the term muMax.
	for (int i = 0; i < N; i++) {
		diff = 0;
		//Compare the difference for points inside the cluster
		for (int l = 0; l < N; l++) {
			if (l != i && dataPoints[l] == dataPoints[i]) {
				for (int j = 0; j < D; j++) {
					diff += abs(inputFile[i][j] - inputFile[l][j]);
					inside++;
				}
			}
		}
		//Calculate the term muMax for data point[i]
		muMax[i] = diff / (inside - 1.0);
	}

	//Finally, calculate the SW index
	for (int i = 0; i < N; i++) {
		if (muMinOut > muMax)
			max = muMinOut[i];
		else
			max = muMax[i];
		SW += ((muMinOut[i] - muMax[i]) / max);
	}
	SW = SW / N;
}


//Calculates the Calinski-Harabasz(CH) internal validity index
void KMeans::calculateCH(int iteration) {

	//Initialization of variables and term vectors
	double traceSB = 0;
	vector <double> mu;
	vector <int> counter;
	vector <vector <double> > muC, muCT;
	//Resize vectors dynamically based on input dataset
	mu.resize(D, 0);
	counter.resize(K, 0);
	muC.resize(K, vector<double>(D, 0));
	muCT.resize(D, vector<double>(K, 0));

	//Iterate through all points in dataset to calculate mu
	for (int j = 0; j < D; j++) {
		for (int i = 0; i < N; i++) {
			mu[j] += inputFile[i][j];
		}
		//Get final mu
		mu[j] = mu[j] / N;
	}
	//Calculate term muC
	for (int l = 0; l < K; l++) {
		for (int j = 0; j < D; j++) {
			counter[l] = 0;
			for (int i = 0; i < N; i++) {
				if (dataPoints[i][l] == 1) {
					muC[l][j] += inputFile[i][j];
					counter[l]++;
				}
			}
			muC[l][j] = (muC[l][j] / counter[l] - mu[j]);
		}
	}
	//Calculate term muCT
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < D; j++) {
			muCT[j][i] = muC[i][j];
		}
	}
	//Get the term traceSB -- sum of the diagonal elements between-cluster scatter matrices.
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < D; j++) {
			if (i == j) {
				traceSB += counter[i] * muC[i][j] * muCT[i][j];
			}
		}
	}
	CH = ((N - (double)K) * traceSB) / ((K - 1.0) * SSE[iteration]);

}


//Calculates the Davies-Bouldin (DB) index
//While CH and SW are maximization indices, DB is a minimization index.
double KMeans::calculateDB() {
	//Initialization of equation term vectors and variables
	double DB = 0;
	vector <int> counter;
	vector <double> max, mu, sigma;
	vector <vector <double> > sigmaC, muC;
	vector <vector < double> > DBindex;

	//Resize vectors dynamically based on input dataset.
	counter.resize(K);
	mu.resize(K);
	max.resize(K);
	sigma.resize(K);
	sigmaC.resize(K, vector<double>(D, 0));
	muC.resize(K, vector<double>(D, 0));
	DBindex.resize(K, vector<double>(K, 0));

	//Calculate the term muC
	for (int l = 0; l < K; l++) {
		for (int j = 0; j < D; j++) {
			counter[l] = 0;
			for (int i = 0; i < N; i++) {
				if (dataPoints[i][l] == 1) {
					muC[l][j] += inputFile[i][j];
					counter[l]++;
				}
			}
			muC[l][j] = muC[l][j] / counter[l];
		}
	}
	//calculate the term mu
	for (int l = 0; l < K; l++) {
		for (int j = 0; j < D; j++) {
			mu[l] += (muC[l][j] * muC[l][j]);
		}
		mu[l] = mu[l] / D;
	}
	//Calculate the term sigmaC
	for (int l = 0; l < K; l++) {
		for (int j = 0; j < D; j++) {
			for (int i = 0; i < N; i++) {
				sigmaC[l][j] += (inputFile[i][j] - muC[l][j]) * (inputFile[i][j] - muC[l][j]);
			}
			sigmaC[l][j] = sqrt(sigmaC[l][j] / counter[l]);
		}
	}
	//Calculate the term sigma
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < D; j++) {
			sigma[i] += sigmaC[i][j];
		}
		sigma[i] = sigma[i] / D;
	}

	//Calculate the term DBindex
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < K; j++) {
			if (j != i) {
				if (mu[i] != mu[j]) {
					DBindex[i][j] = (sigma[i] + sigma[j]) / (mu[i] - mu[j]);
				}
			}
		}
	}

	//Populate the term max
	max[0] = DBindex[0][1];

	for (int i = 0; i < K; i++) {
		for (int j = 1; j < K; j++) {
			max[i] = DBindex[i][0];
		}
	}
	//Calculate the terms max and DB
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < K; j++) {
			if (i != j && DBindex[i][j] > max[i]) {
				max[i] = DBindex[i][j];
			}
		}
		DB += max[i];
	}
	//Get final DB index
	DB = DB / K;

	return DB;
}


//Phase Four: External Validity Measures
//Phase Four implements Rand, Jaccard, and Fowlkes-Mallows external validity indices
//For external validation, we need
//FP -- number of false positives
//TP -- number of true positives
//FN -- number of false negatives
//TN -- number of true negatives
//total = FP+FN+TP+TN
void KMeans::getExternalMeasures() {
	//Reset/Initialize values
	TP = 0;
	FN = 0;
	FP = 0;
	TN = 0;
	contingency.resize(2, vector<double>(2, 0));

	//Get the number of FP/FN/TP/TN
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i != j && j > i) {
				for (int l = 0; l < tC; l++) {
					if (dataPoints[i][l] == 1) {
						if (dataPoints[i][l] == dataPoints[j][l]) {
							//1:True positive -- points are equal and true clusters match
							if (trueK[i] == trueK[j]) {
								TP++;
							}
							//2:False positive -- true clusters do not match
							else {
								FP++;
							}
						}
						else {
							//3: False Negatice -- Clusters match but points are not equal
							if (trueK[i] == trueK[j]) {
								FN++;
							}
							//4: True Negative -- Neither points nor true clusters match
							else {
								TN++;
							}
						}
					}
				}
			}
		}
	}
	//Sum up values to get the total
	sumEX = TP + FP + FN + TN;


}

//Implementation of external validity indices for Phase Four
//All indices will have values between 0 and 1 inclusive.
//Rand external validation
void KMeans::Rand() {
	exRand = (TP + TN) / sumEX;
}
//Jaccard is a well-known index method
//While not the case for this dataset, it is very useful for binary datasets
void KMeans::Jaccard() {
	exJaccard = TP / (TP + FN + FP);
}
//Fowlkes-Mallows index
void KMeans::FM() {
	exFM = TP / sqrt((TP + FN) * (TP + FP));
}