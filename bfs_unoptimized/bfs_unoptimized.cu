#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_DIST 5000

typedef struct graphVertices {
	int startIndex;
	int numberOfNeighbours;
	
} graphVertices;

__global__ void bfs_unoptimized(graphVertices* graphVertice_gpu, int* NeighboursVertices_gpu, int numberOfNodes, int* result_gpu, bool* gpu_done) {
	
	// Thread index
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	int numOfThreads = blockDim.x * gridDim.x;

	// All the threads traverse the adjancy list
	for(int v = 0; v < numberOfNodes; v += numOfThreads) {
		
		// different index for different thread
		int vertex = v + tid;
	
		// check boundary condition
		if(vertex < numberOfNodes) {
		
			// traverse all the neighbouring vertex concurrently
			for(int n=0;n<graphVertice_gpu[vertex].numberOfNeighbours;n++) {
				int neighbour = NeighboursVertices_gpu[graphVertice_gpu[vertex].startIndex + n];
			
				// computation of the cost and traversal
				if (result_gpu[neighbour] > result_gpu[vertex] + 1) {
 					result_gpu[neighbour] = result_gpu[vertex] + 1;
					*gpu_done = 0;									
 
				}
				

			}

		}
	} 
	
	
}

int main( int argc, char* argv[] ) { 
	// input from user
	int NUM_NODES = atoi(argv[1]);

	// kernel parameters
	int block_size = 1024;
	int grid_size = NUM_NODES/block_size;

	// structure declaration
	graphVertices vertice[NUM_NODES];

	// Array of neighbouring nodes
	int neighbourVertices[NUM_NODES];  

	// populate the graph
	for(int i=0;i<NUM_NODES;i++) {
		vertice[i].numberOfNeighbours = 2;// (rand() % 5)+1;
	}
	
	vertice[0].startIndex = 0;
	for(int j=1;j<NUM_NODES;j++) {
		vertice[j].startIndex = vertice[j-1].startIndex + vertice[j-1].numberOfNeighbours;
	}

	for(int k=0;k<NUM_NODES*2;k++) {
		neighbourVertices[k] = k+1;
	}

	int start_vertex = neighbourVertices[1];

	cudaSetDevice(0);
        
	// Time Variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	// Variable declaration for GPU
	graphVertices* graphVertice_gpu;
	int* neighbourVertices_gpu;
	int* result_gpu;
	bool* gpu_done;

	// Memory allocation for GPU variables
	cudaMalloc((void**)&graphVertice_gpu, sizeof(graphVertices)*NUM_NODES);
	cudaMalloc((void**)&neighbourVertices_gpu, sizeof(int)*NUM_NODES*2);
	cudaMalloc((void**)&result_gpu, sizeof(int)*NUM_NODES);
	cudaMalloc((void**)&gpu_done, sizeof(bool) * 1);

	int kernel_call_count;		
	int* result_cpu;
	bool* cpu_done = new bool[1];
	result_cpu = new int[NUM_NODES];

	for(int i=0;i<NUM_NODES;i++) {
		result_cpu[i] = MAX_DIST;
	}

	result_cpu[start_vertex] = 0;

	// Transfer data from CPU to GPU
	cudaMemcpy(result_gpu, result_cpu, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);
	cudaMemcpy(graphVertice_gpu, vertice, sizeof(graphVertices)*NUM_NODES, cudaMemcpyHostToDevice);
	cudaMemcpy(neighbourVertices_gpu, neighbourVertices, sizeof(int)*NUM_NODES*2, cudaMemcpyHostToDevice);

	printf("Running parallel job.\n");

	cudaEventRecord(start,0);
	bool false_value = 1;

	do
	{
		kernel_call_count++;		
		cudaMemcpy(gpu_done, &false_value, sizeof(bool) * 1, cudaMemcpyHostToDevice);

		// call the kernel
		bfs_unoptimized<<<grid_size, block_size>>>(graphVertice_gpu, neighbourVertices_gpu, NUM_NODES, result_gpu, gpu_done);
		
		cudaMemcpy(cpu_done, gpu_done , sizeof(bool) * 1, cudaMemcpyDeviceToHost);
	} while(*cpu_done != 0);

	// Transfer result back from GPU to CPU
	cudaMemcpy(result_cpu, result_gpu, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost);

	printf("Kernel call : %d\n", kernel_call_count);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("Parallel Job Time: %.2f ms\n", time);

	cudaFree(graphVertice_gpu);
	cudaFree(neighbourVertices_gpu);
	cudaFree(result_gpu);
	cudaFree(gpu_done);
	
	return 0;
}

