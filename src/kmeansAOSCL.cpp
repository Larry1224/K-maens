#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include <iostream>

#include <fstream>

#include <vector>
#include <unordered_set>

#include <random>
#include <cstdlib>
#include <cmath>
#include <CL/opencl.hpp>

#include "../inc/kmeans.hpp"

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
auto getPlatform(const std::string &vendorNameFilter)
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	for (const auto &p : platforms)
	{
		if (p.getInfo<CL_PLATFORM_VENDOR>().find(vendorNameFilter) != std::string::npos)
		{
			return p;
		}
	}
	throw cl::Error(CL_INVALID_PLATFORM, "No platform has given vendorName");
}

auto getDevice(cl::Platform &platform, cl_device_type type, size_t globalMemoryMB)
{
	std::vector<cl::Device> devices;
	platform.getDevices(type, &devices);
	globalMemoryMB *= 1024 * 1024; // from MB to bytes
	for (const auto &d : devices)
	{
		if (d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() >= globalMemoryMB)
			return d;
	}
	throw cl::Error(CL_INVALID_DEVICE, "No device has needed global memory size");
}

string src = R"CLC(
	__kernel void distances(__global const float *pts, __global const float *currentCentriods
							, uint N, uint K, uint D, __global float *distance){
		uint which_k = get_global_id(0);
		uint which_pt = get_global_id(1);
		if(which_pt < N && which_k < K){
			float tmp = 0.0f;
			for (uint which_D = 0; which_D < D; which_D++){
				tmp += pow((pts[which_pt * D + which_D]-currentCentriods[which_k * D + which_D]), 2);
			}
			distance[K * which_pt + which_k ] = tmp;
		}
	}
	__kernel void assigngroup(__global const float *distance, __global int *group, uint N, uint K){
		uint which_pt = get_global_id(0);
		if (which_pt <N){
			__global const float *ptr = distance + K*which_pt;
			float min = ptr[0];
			int grp = 0;
			for (uint i = 1; i < K; i++)
			{
				if (ptr[i]<min){
					grp=i;
					min =ptr[i];
				}
			}
			group[which_pt] = grp;
		}
	}
	__kernel
	void myAtomicAddG(__global float *addr, float val) {
		union {
			uint u32;
			float f32;
		} current, expected, next;

		do {
			current.f32 = *addr;
			next.f32 = current.f32 + val;
			expected.u32 = current.u32;
			current.u32 = atomic_cmpxchg( (__global uint*) addr, expected.u32, next.u32 );
		} while( current.u32 != expected.u32 );
	}
	__kernel void sumcenteriod(__global const uint* group, __global uint* groupcount, __global float* centriods
								, __global const float* pts,uint N, uint K, uint D){
		uint which_pt = get_global_id(0);
		if(which_pt >= N) return;
		uint whichGroup = group[which_pt];
		atomic_inc(groupcount + whichGroup);
		for(uint which_D=0; which_D < D; which_D++){
			myAtomicAddG(centriods + whichGroup * D+ which_D, pts[which_pt *D +which_D]);
		}
	}
	__kernel void avgcen(__global float* centeriods, __global int* groupcount, uint K, uint D){
		uint count_k = get_global_id(0);
		if (count_k >= K) return;

		for(uint which_D=0 ; which_D < D ; which_D++ ){
			centeriods[count_k * D+ which_D] /= groupcount[count_k];
		}
	}
	__kernel void hasConverge(__global float *currentCentriods,__global float *oldCentriods,__global uint *dev_result,uint K,uint DIM,float tolerance)
	{
		uint count_k = get_global_id(0);
		if(count_k >= K) return;
		float final_distance = 0.0f;
		for (uint whichDim = 0; whichDim < DIM; whichDim++){
			final_distance += pow((currentCentriods[DIM * count_k + whichDim] - oldCentriods[DIM * count_k + whichDim]), 2);
		}
		if (final_distance > tolerance) {
			atomic_inc(dev_result);
		}
	}
)CLC";

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
			while (*it != ',')
				++it;
			data.pts[whichPt * data.DIM + whichDim] = x;
			it++;
		}
	}
	inp.close();
	//platfrom
	data.platform = getPlatform("NVIDIA");
	data.device = getDevice(data.platform, CL_DEVICE_TYPE_GPU, 1024);
	data.ctx = cl::Context(data.device);
	data.queue = cl::CommandQueue(data.ctx, data.device);
	//Buffer
	data.dev_pts = cl::Buffer(data.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
							  noLines * noAttributes * sizeof(float), data.pts.data());
	data.prg = cl::Program(data.ctx, src);
	try
	{
		data.prg.build();
	}
	catch (cl::Error &e)
	{
		std::cerr << data.prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(data.device);
		throw cl::Error(CL_INVALID_PROGRAM, "Failed to build kennel");
	}
	return true;
}
// AOS version
// array of structs x1, y1, z1, x2, y2, z2, ...
void InitializeCentroid(myData &data, size_t noClusters)
{
	data.K = noClusters;
	data.dev_currentCentroids = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, data.K * data.DIM * sizeof(float));
	data.currentCentroids = vector<float>(data.K * data.DIM);
	size_t tmp[3] = {0, 1, 2};
	// size_t tmp;
	// random_device rd;
	// default_random_engine generator = default_random_engine(rd());
	// uniform_int_distribution<size_t> dis(0, data.N - 1);
	for (size_t i = 0; i < data.K; i++)
	{
		// tmp = dis(generator);
		for (size_t j = 0; j < data.DIM; j++)
		{
			data.currentCentroids[i * data.DIM + j] = data.pts[tmp[i] * data.DIM + j];
		}
	}
	data.queue.enqueueWriteBuffer(data.dev_currentCentroids, CL_TRUE, 0, data.K * data.DIM * sizeof(float), data.currentCentroids.data());
}

void AssignGroups(myData &data)
{
	data.group = vector<uint>(data.N);
	static cl::Buffer distance(data.ctx, CL_MEM_READ_WRITE, data.N * data.K * sizeof(float));
	static cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl_uint, cl_uint, cl_uint, cl::Buffer &> distancekennel(data.prg, "distances");
	static cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl_uint, cl_uint> groupkernel(data.prg, "assigngroup");
	auto config = cl::EnqueueArgs(data.queue, {(data.K + 15) / 16 * 16, (data.N + 15) / 16 * 16}, {16, 16});
	distancekennel(config, data.dev_pts, data.dev_currentCentroids, data.N, data.K, data.DIM, distance);
	data.dev_group = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, data.N * sizeof(cl_int));
	auto config1 = cl::EnqueueArgs(data.queue, (data.N + 255) / 256 * 256, 256);
	groupkernel(config1, distance, data.dev_group, data.N, data.K);
}

void UpdateCentroids(myData &data)
{
	data.oldCentroids = vector<float>(data.K * data.DIM);
	data.dev_oldCentriods = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, data.K * data.DIM * sizeof(float));
	swap(data.dev_currentCentroids, data.dev_oldCentriods);
	data.queue.enqueueFillBuffer(data.dev_currentCentroids, 0, 0, data.K * data.DIM * sizeof(float));
	auto groupcount = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, data.K * sizeof(cl_uint));
	data.queue.enqueueFillBuffer(groupcount, 0, 0, data.K * sizeof(cl_uint));

	static cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl_uint, cl_uint, cl_uint> sumcenkernel(data.prg, "sumcenteriod");
	static cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl_uint, cl_uint> avgcenkernel(data.prg, "avgcen");
	auto config = cl::EnqueueArgs(data.queue, (data.N + 255) / 256 * 256, 256);
	sumcenkernel(config, data.dev_group, groupcount, data.dev_currentCentroids, data.dev_pts, data.N, data.K, data.DIM);
	auto config1 = cl::EnqueueArgs(data.queue, (data.K + 255) / 256 * 256, 256);
	avgcenkernel(config1, data.dev_currentCentroids, groupcount, data.K, data.DIM);
}

bool HasConverged(myData &data, float tolerance)
{
	uint result = 0;
	tolerance *= tolerance;
	static cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl_uint, cl_uint, cl_float> hasConvergekernel(data.prg, "hasConverge");
	static auto config = cl::EnqueueArgs(data.queue, (data.K + 255) / 256 * 256, 256);
	static auto dev_result = cl::Buffer(data.ctx, CL_MEM_READ_WRITE, sizeof(cl_uint));
	data.queue.enqueueFillBuffer(dev_result, 0, 0, sizeof(cl_uint));
	hasConvergekernel(config, data.dev_currentCentroids, data.dev_oldCentriods, dev_result, data.K, data.DIM, tolerance);
	data.queue.enqueueReadBuffer(dev_result, CL_TRUE, 0, sizeof(cl_uint), &result);

	return !result;
}
