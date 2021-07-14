#include "../inc/kmeans.hpp"
#include <iostream>
#include <fstream>

#include <vector>
#include <unordered_set>

#include <random>
#include <cstdlib>
#include <cmath>
using namespace std;

/*
     1) N: the number of points, size_t
     2) DIM: the dimensionality, size_t
     3) pts: coordinates of each data-point, float[N*DIM]
     4) K: the number of clusters, size_t
     5) cCentroid: coordinates of k current centroids, float[k*DIM]
     6) pCentroid: coordinates of k previosu centroids, float[k*DIM]
     7) group: the assignment of each point to a cluster, size_t[N]
*/

bool readCSV(myData &data, const char *filename)
{
	ifstream inp(filename);
	if (!inp)
	{
		return false;
	}

	char buf[4096];
	// let's read the first line and determine how many attributes are there
	inp.getline(buf, 4096);
	size_t noAttributes = 0;
	for (auto it = buf; *it != 0; ++it)
	{
		if (*it == ',')
			++noAttributes;
	}

	// Then let's count how many more lines are there.
	size_t noLines = 1;
	while (inp.good())
	{
		inp.getline(buf, 4096);
		if (inp.gcount() > 0)
			noLines++;
	}
	// cout << noLines << ", " << noAttributes << endl;
	// the number of comma is the number of data
	// noLines is the number of points, and noAttributes is the dimension for each point, they should be stored into your data.
	// [!] You should now allocate memory for storing point data!!

	// re-read the file and this time read the actual data into allocated memory
	inp.clear();
	inp.seekg(0, inp.beg);
	for (size_t whichPt = 0; whichPt < noLines; ++whichPt)
	{
		inp.getline(buf, 4096);
		auto it = buf;
		for (size_t whichDim = 0; whichDim < noAttributes; ++whichDim)
		{
			auto x = atof(it);
			// x is the coordinate of the dimension which Dim of the point whichPt, and you should store it somehow.
			// e.g. data.pts[whichPt][whichDim] = x;
			// [!] You should add some code here to store read data x into your data...
			while (*it != ',')
				++it;
			it++;
		}
	}
	inp.close();
	return true;
}

// AOS version
// array of structs x1, y1, z1, x2, y2, z2, ...
void InitializeCentroid(myData &data, size_t noClusters)
{
	// [!] We finally know how many clusters we are going to form.  You should do some memory allocation here.
	cout << "\nDo remember to implement InitializeCentroid";
}

void AssignGroups(myData &data)
{
	cout << "\nDo remember to implement InitializeCentroid";
}

void UpdateCentroids(myData &data)
{
	// copy the current centroids into oldCentroids
	cout << "\nDo remember to implement UpdateCentroids.";
}

bool HasConverged(myData &data, const float tolerance)
{
	cout << "\nDo remember to implement HasConverged.";
	return false;
}
#include "../inc/kmeans.hpp"
#include <iostream>
#include <fstream>

#include <vector>
#include <unordered_set>

#include <random>
#include <cstdlib>
#include <cmath>
using namespace std;

/*
     1) N: the number of points, size_t
     2) DIM: the dimensionality, size_t
     3) pts: coordinates of each data-point, float[N*DIM]
     4) K: the number of clusters, size_t
     5) cCentroid: coordinates of k current centroids, float[k*DIM]
     6) pCentroid: coordinates of k previosu centroids, float[k*DIM]
     7) group: the assignment of each point to a cluster, size_t[N]
*/

bool readCSV(myData &data, const char *filename)
{
	ifstream inp(filename);
	if (!inp)
	{
		return false;
	}

	// let's count # lines first.
	size_t noLines = 1;
	char buf[4096];
	// the number of comma is the number of data
	size_t noAttributes = 0;
	inp.getline(buf, 4096);
	for (auto it = buf; *it != 0; ++it)
	{
		if (*it == ',')
			++noAttributes;
	}
	while (inp.good())
	{
		inp.getline(buf, 4096);
		if (inp.gcount() > 0)
			noLines++;
	}
	// cout << noLines << "***" << endl;
	data.N = noLines;
	data.DIM = noAttributes;
	data.pts = vector<float>(data.N * data.DIM);
	// noLines is the number of points, and noAttributes is the dimension for each point, they should be stored into your data.
	// [!] You should now allocate memory for storing point data!!

	// re-read the file and this time read the actual data into allocated memory
	inp.clear();
	inp.seekg(0, inp.beg);
	for (size_t whichPt = 0; whichPt < noLines; ++whichPt)
	{
		inp.getline(buf, 4096);
		auto it = buf;
		for (size_t whichDim = 0; whichDim < noAttributes; ++whichDim)
		{
			auto x = atof(it);
			// x is the coordinate of the dimension which Dim of the point whichPt, and you should store it somehow.
			// e.g. data.pts[whichPt][whichDim] = x;
			// [!] You should add some code here to store read data x into your data...
			//data.pts[whichPt*data.N+whichDim]=x;
			while (*it != ',')
				++it;
			data.pts[whichDim * data.N + whichPt] = x;
			it++;
		}
	}
	inp.close();
	// for (float str: data.pts)
	//     cout << str << endl;
	//test data in the vector
	return true;
}

// AOS version
// array of structs x1, y1, z1, x2, y2, z2, ...
void InitializeCentroid(myData &data, size_t noClusters)
{
	// [!] We finally know how many clusters we are going to form.  You should do some memory allocation here.
	//cout << "\nDo remember to implement InitializeCentroid";
	data.K = noClusters;
	data.currentCentroids = vector<float>(data.K * data.DIM);
	size_t tmp;
	random_device rd;
	default_random_engine generator = default_random_engine(rd());
	uniform_int_distribution<size_t> dis(0, data.N - 1);
	for (size_t i = 0; i < data.K; i++)
	{
		tmp = dis(generator);
		for (size_t j = 0; j < data.DIM; j++)
		{
			data.currentCentroids[j * data.K + i] = data.pts[j * data.N + tmp];
		}
	}
	// for (float str: data.currentCentroids)
	//     cout << str << endl;
}

void AssignGroups(myData &data)
{
	//cout << "\nDo remember to implement InitializeCentroid";
	data.group = vector<size_t>(data.N);
	float tmp = 0.0;
	float tmp2 = 0.0;
	vector<float> tmp1 = vector<float>(data.N * data.K);
#pragma omp parallel reduction(+ \
							   : tmp)
	{
#pragma omp for schedule(dynamic, 32) collapse(2)
		for (size_t i = 0; i < data.K; i++)
		{
			for (size_t j = 0; j < data.N; j++)
			{
				tmp = 0.0;
				// #pragma omp parallel for schedule(guided, 32) 會導致執行序長時間滿載
				for (size_t k = 0; k < data.DIM; k++)
				{
					tmp += pow((data.pts[k * data.N + j] - data.currentCentroids[k * data.K + i]), 2);
				}
				tmp1[i * data.N + j] = sqrt(tmp);
			}
		}
	}
	for (size_t i = 0; i < data.N; i++)
	{
		tmp2 = tmp1[i];
		for (size_t j = 0; j < data.K; j++)
		{
			if (tmp2 >= (tmp1[i + data.N * j]))
			{
				tmp2 = tmp1[i + data.N * j];
				data.group[i] = j;
			}
		}
	}
	// for (float str: tmp1)
	//     cout << str << endl;
	// for (float str: data.group)
	//     cout << str << endl;
}

void UpdateCentroids(myData &data)
{
	// copy the current centroids into oldCentroids
	//cout << "\nDo remember to implement UpdateCentroids.";
	data.oldCentroids = vector<float>(data.K * data.DIM);
	size_t count = 0;
	for (size_t i = 0; i < data.K * data.DIM; i++)
	{
		data.oldCentroids[i] = data.currentCentroids[i];
		data.currentCentroids[i] = 0;
	}
	for (size_t i = 0; i < data.K; i++)
	{
		for (size_t j = 0; j < data.DIM; j++)
		{
			for (size_t k = 0; k < data.N; k++)
			{
				if (data.group[k] == i)
				{
					data.currentCentroids[j * data.K + i] += data.pts[j * data.N + k];
					count++;
				}
			}
			data.currentCentroids[j * data.K + i] = data.currentCentroids[j * data.K + i] / count;
			count = 0;
		}
	}
	// for (float str: data.currentCentroids)
	//     cout << str << endl;
}

bool HasConverged(myData &data, const float tolerance)
{
	//cout << "\nDo remember to implement HasConverged.";
	vector<float> tmp = vector<float>(data.K);
	float tmp1;
	for (size_t i = 0; i < data.K; i++)
	{
		tmp1 = 0.0;
#pragma omp parellel for schedule(dynamic, 32)
		for (size_t j = 0; j < data.DIM; j++)
		{
			tmp1 += pow((data.currentCentroids[j * data.K + i] - data.oldCentroids[j * data.K + i]), 2);
		}
		tmp[i] = sqrt(tmp1);
		if (tmp[i] > tolerance)
		{
			return false;
		}
	}
	// for (float str: tmp)
	//     cout << str << endl;
	return true;
}
