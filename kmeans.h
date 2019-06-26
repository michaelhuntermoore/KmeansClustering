#ifndef KMEANS_H__
#define KMEANS_H__

#include <map>
#include <string>
#include <vector>
using namespace std;
//Define a class for the Kmeans clusters
class KMeans {

	//2D vectors for storage of data points and centroids
	//For Phase Four, a vector to store contingency is added
	vector <vector <double>> dataPoints, inputFile, centroids,nearest, contingency;
	//Vector the store the squared-error for each centroid
	vector <double> SE;
	//Input parameters and file parameters
	int K,		//Input number of clusters
		N,		//Number of data points in the input file
		D,		//Dimensionality of the input file
		I;	//Input number of iterations
	double T;		//Input convergence threshold
	double maxSE; // maximum squared-error
	double trueCluster; //Temp storage of true clusters.

public:
	//Constructor for KMeans
	KMeans(int K, int N, int D, int I, double T, vector <vector <double> > data);
	void updatePoints();
	void calculateCentroids(int iteration);
	//For Phase Two, Maximin and Random Partition methods have been added
	//In addition to the typical Random Selection method.
	void randomPartition();
	void maxiMin();
	
	//For Phase Three, three internal validity indices have been added.
	void calculateSW();
	void calculateCH(int iteration);
	double calculateDB();
	
	//For Phase Four, three external validity indices have been added.
	void getExternalMeasures(); //Acquire TP/TN/FP/FN for the clustered dataset
	void Rand(); //Rand External index
	void Jaccard(); //Jaccard External index
	void FM(); //Fowlkes-Madllows External index

	vector <double> SSE, maxSSE, sumK, trueK;
	//For Phase Three, variables for internal indices have been added.
	double SW = 0, CH = 0, DB = 0;

	//For Phase Four, variables for external validation have been added.
	double FP = 0, FN = 0, TP = 0, TN = 0;
	double sumEX = 0;
	double exRand = 0, exJaccard = 0, exFM = 0;
	double tC = 0, kMin = 0, kMax = 0;
};

#endif  //KMEANS_H__