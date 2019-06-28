
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector> 
#include <algorithm> 

#define SEQ_SIZE 9
#define PATTERN_SIZE 4

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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

int variations_without_repetitions_count(int n, int k) {
	if (k > n) {
		return 1;
	}

	int result = 1;
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

	auto patternValues = getUniqueValues(&pattern);
	auto seqValues = getUniqueValues(&seq);

	std::cout << variations_without_repetitions_count(12, 15);

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