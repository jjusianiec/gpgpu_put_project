
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector> 
#include <algorithm> 

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define SEQ_SIZE 9
#define PATTERN_SIZE 4


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void multiplyBy2(int* data, unsigned int n) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; 
	if (tid < n) { 
		data[tid] = 2 * data[tid]; 
	} 
}

template<typename T>
std::vector<T>* getUniqueValues(std::vector<T>* input) {
	std::vector<T>* uniqueValues = new std::vector<T>(*input);
	std::sort(uniqueValues->begin(), uniqueValues->end());
	auto ip = std::unique(uniqueValues->begin(), uniqueValues->end());
	auto begin = uniqueValues->begin();
	uniqueValues->resize(std::distance(begin, ip));
	return uniqueValues;
}

template<typename T>
thrust::host_vector<T>* getHostVector(std::vector<T>* input) {
	thrust::host_vector<T>* host_vector = new thrust::host_vector<T>();
	for (auto it = input->begin(); it != input->end(); ++it) {
		host_vector->push_back(*it);
	}
	return host_vector;
}

unsigned long long variations_without_repetitions_count(int n, int k) {
	if (k > n) {
		return 1;
	}

	unsigned long long result = 1;
	for (int i = n; i > n - k; i--) {
		result *= i;
	}

	return result;
}


int main()
{  
	// 124353621
    //const int seq[SEQ_SIZE] = { 1,2, 4, 3, 5, 3, 6, 2, 1 };
    //const char pattern[PATTERN_SIZE] = { 'a', 'b', 'b', 'a' };
	std::vector<int> seq = { 1,2, 4, 3, 5, 3, 6, 2, 1 };
	std::vector<char> pattern = { 'a', 'b', 'b', 'a' };

	thrust::host_vector<char>* patternValues = getHostVector(getUniqueValues(&pattern));
	thrust::host_vector<int>* seqValues = getHostVector(getUniqueValues(&seq));

	thrust::device_vector<char>* devicePatternValues = new thrust::device_vector<char>();
	thrust::device_vector<int>* deviceSeqValues = new thrust::device_vector<int>();

	deviceSeqValues->resize(seqValues->size());

	*deviceSeqValues = *seqValues;

	multiplyBy2 <<< 1, 10 >>> (deviceSeqValues->data().get(), deviceSeqValues->size());

	*seqValues = *deviceSeqValues;

	for (auto it = seqValues->begin(); it != seqValues->end(); ++it) {
		std::cout << *it;
	}

	//std::cout << variations_without_repetitions_count(15, 12);

	free(seqValues);
	free(patternValues);
    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}