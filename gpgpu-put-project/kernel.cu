
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector> 
#include <algorithm> 

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#define BLOCK_SIZE 1024


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

__device__ void variations_without_repetitions_count_dev(int n, int k, unsigned long long* result) {
	if (k > n) {
		*result = 1;
		return;
	}

	*result = 1;
	for (int i = n; i > n - k; i--) {
		*result *= i;
	}
}

__device__ void variation(int n, int k, int variationNumber, int* result) {
	bool* isTaken = new bool[n];
	for (int i = 0; i < n; i++) {
		isTaken[i] = false;
	}
	for (int x = 0; x < k ; x++) {
		unsigned long long v = 0;
		variations_without_repetitions_count_dev(n - x - 1, k - x - 1, &v);
		auto t = variationNumber / v;
		int searchedPosition = -1;
		int realPosition = 0;
		for (int i = 0; i < n; i++) {
			if (!isTaken[i]) {
				searchedPosition++;
				if (t == searchedPosition) {
					realPosition = i;
					break;
				}
			}
			
		}
		isTaken[realPosition] = true;
		result[x] = realPosition;
		variationNumber %= v;
	}
}


__global__ void findSubstitution(
	char* patternValues, int patternValuesSize, 
	int* seqValues, int seqValuesSize, 
	char* pattern, int patternSize,
	int* seq, int seqSize,
	int* result, unsigned long long variationCount) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > variationCount) return;
	
	int* variationResult = new int[patternValuesSize];
	variation(seqValuesSize, patternValuesSize, index, variationResult);
}

int main()
{  
	// 124353621
    //const int seq[SEQ_SIZE] = { 1,2, 4, 3, 5, 3, 6, 2, 1 };
    //const char pattern[PATTERN_SIZE] = { 'a', 'b', 'b', 'a' };
	std::vector<int> seq = { 1,2, 4, 3, 5, 3, 6, 2, 1 };
	std::vector<char> pattern = { 'a', 'b', 'b', 'a' };

	thrust::host_vector<char>* patternValues = getHostVector(getUniqueValues(&pattern));
	thrust::host_vector<char>* thrustPattern = getHostVector(&pattern);
	thrust::host_vector<int>* seqValues = getHostVector(getUniqueValues(&seq));
	thrust::host_vector<int>* thrustSeq = getHostVector(&seq);

	thrust::device_vector<char>* devPatternValues = new thrust::device_vector<char>();
	thrust::device_vector<char>* devThrustPattern = new thrust::device_vector<char>();
	thrust::device_vector<int>* devSeqValues = new thrust::device_vector<int>();
	thrust::device_vector<int>* devThrustSeq = new thrust::device_vector<int>();

	//for (int i = 0; i < variations_without_repetitions_count(seqValues->size(), patternValues->size()); i++) {
	//	int* result = new int[patternValues->size()];
	//	variation(seqValues->size(), patternValues->size(), i, result);
	//	for (int j = 0; j < patternValues->size(); j++) {
	//		//std::cout << (*seqValues)[result[j]];
	//		std::cout << result[j];
	//	}
	//	std::cout << std::endl;
	//}

	unsigned long long variationCount = variations_without_repetitions_count(seqValues->size(), patternValues->size());
	int gridSize = variationCount / BLOCK_SIZE;
	if (gridSize < 1) {
		gridSize = 1;
	}
	int* result = new int[variationCount];

	devPatternValues->resize(patternValues->size());
	devThrustPattern->resize(thrustPattern->size());
	devSeqValues->resize(seqValues->size());
	devThrustSeq->resize(thrustSeq->size());

	*devPatternValues = *patternValues;
	*devThrustPattern = *thrustPattern;
	*devSeqValues = *seqValues;
	*devThrustSeq = *thrustSeq;

	findSubstitution <<< gridSize, BLOCK_SIZE >>> (
		devPatternValues->data().get(), devPatternValues->size(),
		devSeqValues->data().get(), devSeqValues->size(),
		devThrustPattern->data().get(), devThrustPattern->size(),
		devThrustSeq->data().get(), devThrustSeq->size(),
		result, variationCount);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "cuda error: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}

	//thrust::device_vector<char>* devicePatternValues = new thrust::device_vector<char>();
	//thrust::device_vector<int>* deviceSeqValues = new thrust::device_vector<int>();

	//deviceSeqValues->resize(seqValues->size());

	//*deviceSeqValues = *seqValues;

	//multiplyBy2 <<< 1, 10 >>> (deviceSeqValues->data().get(), deviceSeqValues->size());
	//cudaError_t err = cudaGetLastError();
	//if (err != cudaSuccess) {
	//	std::cout << "cuda error: " << cudaGetErrorString(err) << std::endl;
	//	return 1;
	//}

	//*seqValues = *deviceSeqValues;

	//for (auto it = seqValues->begin(); it != seqValues->end(); ++it) {
	//	std::cout << *it;
	//}



	//std::cout << variations_without_repetitions_count(seqValues->size(), patternValues->size()) << std::endl;
	//std::cout << variations_without_repetitions_count(4, 3);

	//free(seqValues);
	//free(patternValues);
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