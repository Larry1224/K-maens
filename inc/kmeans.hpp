#pragma once
#include <vector>
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

using namespace std;
using std::vector;
/* 
The struct myData stores:
     1) N: the number of points, size_t
     2) DIM: the dimensionality, size_t
     3) pts: coordinates of each data-point, float[N*DIM]
     4) K: the number of clusters, size_t
     5) cCentroid: coordinates of k current centroids, float[k*DIM]
     6) pCentroid: coordinates of k previosu centroids, float[k*DIM]
     7) group: the assignment of each point to a cluster, size_t[N]
*/
struct myData
{
     size_t DIM;
     size_t N;
     size_t K;
     vector<float> pts;
     vector<float> currentCentroids;
     vector<float> oldCentroids;
     // vector<uint> group;
     vector<size_t> group;

     cl::Platform platform;
     cl::Device device;
     cl::Context ctx;
     cl::CommandQueue queue;
     cl::Program prg;
     cl::Buffer dev_pts, dev_currentCentroids, dev_oldCentriods, dev_group;
};

// 0. Read CSV file and save data into myData.
// returns false if something went wrong; otherwise returns true.
bool readCSV(myData &, const char *filename);

// 1. initializes K randomly selected centroids, and you should set K to noClusters
//    and allocate memory for storing centroids.
void InitializeCentroid(myData &, size_t noClusters);

// 2. for each point, find the nearest centroid and assign to that group
void AssignGroups(myData &);

// 3. copy data in cCentroid into pCentroid, and then update centroids based on the kmeans algorithm.
void UpdateCentroids(myData &);

// 4. find the centroid that moves furtherest, and if its moving distance is less than the given tolerance, then we say it has converged.
bool HasConverged(myData &, const float tolerance = 1.0e-6);
