#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_NODES 1024

// Declaration of a structure 
typedef struct {
	int startIndex; // starting index in Adj list	
	int numberOfNeighbors; // number of neighbors of each vertices
} Node;

__global__ void bfs_optimized(Node *gpu_vertex, int *gpu_neighbors, bool *gpu_frontier, bool *gpu_visited, int *gpu_cost, bool *gpu_done) {

	// ThreadID
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	
	// boundary condition for threadID
	if (threadId > NUM_NODES)
		*gpu_done = false;
	
	// checking condition for frontier and visited node array
	if (gpu_frontier[threadId] == true && gpu_visited[threadId] == false) { 
		
		// Init	
		gpu_frontier[threadId] = false;
		gpu_visited[threadId] = true;
	
		// assign values from array
		int startPoint = gpu_vertex[threadId].startIndex;
		int endPoint = startPoint + gpu_vertex[threadId].numberOfNeighbors;

		// traverse to the neighbors for every vertex
		for (int i = startPoint; i < endPoint; i++) {
			int neighbor = gpu_neighbors[i];

			// check visited mark and increase cost
			if (gpu_visited[neighbor] == false) {
				gpu_cost[neighbor] = gpu_cost[threadId] + 1;
				gpu_frontier[neighbor] = true;
				*gpu_done = false;

			}

		}

	}

}

// Main method
int main(int argc, char* argv[]) {

	// Kernel launch parameters
        int numberOfThreads = 1024;
	int numberOfBlocks = NUM_NODES/numberOfThreads;

	// Intialization of struct and neighbors array
	Node vertex[NUM_NODES];
	int edges[NUM_NODES];

	// populate the graph
        for(int i=0;i<NUM_NODES;i++) {
                vertex[i].numberOfNeighbors = 1;//(rand() % 5)+1;
        }

        vertex[0].startIndex = 0;
        for(int j=1;j<NUM_NODES;j++) {
                vertex[j].startIndex = vertex[j-1].startIndex + vertex[j-1].numberOfNeighbors;
        }

 	for(int k=0;k<NUM_NODES;k++) {
                edges[k] = k+1;
        }

	cudaSetDevice(0);

	// Time variable
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Intitalization of array for frontier and visited nodes and costpath
	bool frontierArray[NUM_NODES] = { false };
	bool visitedNodes[NUM_NODES] = { false };
	int costOfPath[NUM_NODES] = { 0 };

	int source = 0;
	frontierArray[source] = true;

	// GPU variable declaration
	Node* gpu_vertex;
	int* gpu_neighbors;
	bool* gpu_frontier;
	bool* gpu_visited;
	int* gpu_cost;
	bool* gpu_done;

	// GPU memory allocation
	cudaMalloc((void**)&gpu_vertex, sizeof(Node)*NUM_NODES);
	cudaMalloc((void**)&gpu_neighbors, sizeof(Node)*NUM_NODES);
	cudaMalloc((void**)&gpu_frontier, sizeof(bool)*NUM_NODES);
	cudaMalloc((void**)&gpu_visited, sizeof(bool)*NUM_NODES);
	cudaMalloc((void**)&gpu_cost, sizeof(int)*NUM_NODES);
	cudaMalloc((void**)&gpu_done, sizeof(bool));

	// Transfer of data from CPU to GPU
	cudaMemcpy(gpu_vertex, vertex, sizeof(Node)*NUM_NODES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_neighbors, edges, sizeof(Node)*NUM_NODES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_frontier, frontierArray, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_visited, visitedNodes, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_cost, costOfPath, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);

	bool cpu_done;

	cudaEventRecord(start, 0);
	int Kernel_call_count = 0;

	do {
		Kernel_call_count++;
		cpu_done = true;
		cudaMemcpy(gpu_done, &cpu_done, sizeof(bool), cudaMemcpyHostToDevice);
	
		// Kernel call
		bfs_optimized<<<numberOfBlocks, numberOfThreads>>>(gpu_vertex, gpu_neighbors, gpu_frontier, gpu_visited, gpu_cost, gpu_done);

		cudaMemcpy(&cpu_done, gpu_done , sizeof(bool), cudaMemcpyDeviceToHost);

	} while (!cpu_done);

	// Copy final results from GPU to CPU
	cudaMemcpy(costOfPath, gpu_cost, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost);
	
	printf("Kernel call count: %d\n", Kernel_call_count);

	cudaEventRecord(stop, 0);

	cudaEventElapsedTime(&time, start, stop);
	printf("Parallel Job execution time: %.2f ms\n", time);

	cudaFree(gpu_vertex);
	cudaFree(gpu_neighbors);
	cudaFree(gpu_frontier);
	cudaFree(gpu_visited);
	cudaFree(gpu_cost);
	cudaFree(gpu_done);

	return 0;
}
