#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define TIMER_CREATE(t)             \
	cudaEvent_t t##_start, t##_end;     \
	cudaEventCreate(&t##_start);        \
	cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
	cudaEventRecord(t##_start);         \
	cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
	cudaEventRecord(t##_end);                      \
	cudaEventSynchronize(t##_end);                 \
	cudaEventElapsedTime(&t, t##_start, t##_end);  \
	cudaEventDestroy(t##_start);                   \
	cudaEventDestroy(t##_end);     
  

//Function to check for errors
inline cudaError_t checkCuda(cudaError_t result) 
{
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
	return result;
}

//Number of Vertices
#define vertices 5000 

//Number of Edges per Vertex
#define Edge_per_node 4500

//Used to define the weight of each edge
#define Maximum_weight 5

//Setting a value to infinity
#define infinity 10000000

//Kernel Call to initialize the values of the node weights to infinity except for the Source Node which is set to 0
//We mark the source Node to be settled after that and all other nodes unsettled
__global__ void Initializing(int *node_weight_array, int *mask_array, int Source) // CUDA kernel
{
	int id = blockIdx.x*blockDim.x+threadIdx.x; // Get global thread ID
	if(id<vertices)
	{
		if(id==Source)
		{
			node_weight_array[id]=0;
			mask_array[id]=0;
		}
		else
		{
			node_weight_array[id]=infinity;
			mask_array[id]=0;
		}
	}
}

//Kernel Call to choose the node with  minimum node weight whose edges are to be relaxed
__global__ void Minimum(int *mask_array,int *vertex_array,int *node_weight_array, int *edge_array, int *edge_weight_array, int *min)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x; // Get global thread ID
	if(id<vertices)
	{
		if(mask_array[id]!=1 && node_weight_array[id]<infinity)
		{
			atomicMin(&min[0],node_weight_array[id]);
		}
	}		
}

//Kernel call to relax all the edges of a node
__global__ void Relax(int *mask_array,int *vertex_array,int *node_weight_array, int *edge_array, int *edge_weight_array, int *min)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x; // Get global thread ID
	
	//Iterative variable
	int m,n;

	if(id<vertices)
	{
		if(mask_array[id]!=1 && node_weight_array[id]==min[0])
		{
			mask_array[id]=1;
			for(m=id*Edge_per_node;m<id*Edge_per_node+Edge_per_node;m++)	
			{
				n=edge_array[m];
				atomicMin(&node_weight_array[n],node_weight_array[id]+edge_weight_array[m]);
			}
		}
	}	
}


int main( int argc, char* argv[] )
{

	//Size of the Vertex array
	size_t vertex_array_size = vertices*sizeof(int);
	
	//Size of the edge array and edge_weight array
	size_t edge_array_size = vertices*Edge_per_node*sizeof(int);

	//Intializing the vertex array
	int *vertex_array = (int*)malloc(vertex_array_size); 

	//Initializing a copy of the vertex array
	int *vertex_copy = (int*)malloc(vertex_array_size); 

	//Intializing the edge array
	int *edge_array=(int*)malloc(edge_array_size);

	//Initializing edge_weight_array which stores the weights of each edge
	int *edge_weight_array = (int*)malloc(edge_array_size);

	//Initializing Node weight array which stores the value for the current weight to reach the node
	int *node_weight_array = (int*)malloc(vertex_array_size);

	//Array to mark if a node is settled or not
	int *mask_array = (int*)malloc(vertex_array_size);

	//Iterative operator
	int i,j,k;  

	printf("Initializing Verte Array...\n");

	//Setting node number in vertex_array
	for(i=0;i<vertices;i++)
	{
		vertex_array[i]=i;
	}
	
	//Setting the RNG seed to system clock
	srand(time(NULL));

	//temp variable
	int temp;

	printf("Initializing Edge Array...\n");
	
	//Adding random edges to each node
	memcpy(vertex_copy,vertex_array,vertex_array_size);
	for(i=0;i<vertices;i++)
	{
		for(j=vertices-1;j>0;j--)
		{		
			k=rand()%(j+1);
			temp = vertex_copy[j];
			vertex_copy[j]=vertex_copy[k];
			vertex_copy[k]=temp;
		}

		for(j=0;j<Edge_per_node;j++)
		{
			if(vertex_copy[j]==i)
			{
				j=j+1;
				edge_array[i*Edge_per_node+(j-1)]= vertex_copy[j];			
			}
			else
			{
				edge_array[i*Edge_per_node+j]= vertex_copy[j];			
			}
		}

	}

/*	
	//Can be uncommented to see the edges of each node
	printf("=== Initial edges===\n");
	for(i=0;i<vertices*Edge_per_node;i++)
	{
		printf("E[%d]= %d\n",i,edge_array[i]);
	}
*/	

	printf("Initializing weights of each edge...\n");

	//Adding weights to the edge_weight array
	for(i=0;i<vertices;i++)
	{
		int a = rand()%Maximum_weight+1;
		int b = rand()%Maximum_weight+1;
		for(j=0;j<Edge_per_node;j++)
		{
			edge_weight_array[i*Edge_per_node+j]=a+j*b;
		}
	}

/*	
	//Can be uncommented to check the weight of each edge
	printf("=== Initial edge weight weight===\n");
	for(i=0;i<vertices*Edge_per_node;i++)
	{
		printf("W[%d]= %d\n",i,edge_weight_array[i]);
	}
*/

	//Initializing gpu variables
	int *gpu_vertex_array;
	int *gpu_edge_array;
	int *gpu_edge_weight_array;
	int *gpu_node_weight_array;
	int *gpu_mask_array;

	checkCuda(cudaMalloc(&gpu_vertex_array,vertex_array_size));
	checkCuda(cudaMalloc(&gpu_node_weight_array,vertex_array_size));
	checkCuda(cudaMalloc(&gpu_mask_array,vertex_array_size));
	checkCuda(cudaMalloc(&gpu_edge_array,edge_array_size));
	checkCuda(cudaMalloc(&gpu_edge_weight_array,edge_array_size));

	//Copying memory from Host to Device	
	checkCuda(cudaMemcpy(gpu_vertex_array,vertex_array,vertex_array_size,cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu_node_weight_array,node_weight_array,vertex_array_size,cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu_mask_array,mask_array,vertex_array_size,cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu_edge_array,edge_array,edge_array_size,cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu_edge_weight_array,edge_weight_array,edge_array_size,cudaMemcpyHostToDevice));

	//Declaring the block and grid size
	int blockSize, gridSize;
	blockSize=1024;
	gridSize = (int)ceil((float)vertices/blockSize); // Number of thread blocks in grid

	//Start Timer
	float start_time;
	TIMER_CREATE(start_time);
	TIMER_START(start_time);


	//Kernel Call for initializating of the node weights array
	Initializing<<<gridSize, blockSize>>>(gpu_node_weight_array,gpu_mask_array, 0);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) checkCuda(cudaMemcpy(node_weight_array,gpu_node_weight_array,vertex_array_size,cudaMemcpyDeviceToHost));
	{
		printf("Error: %s\n", cudaGetErrorString(err));
	}

/*	
	//Can be Uncommented too check the initial node weight of each node
	checkCuda(cudaMemcpy(node_weight_array,gpu_node_weight_array,vertex_array_size,cudaMemcpyDeviceToHost));	
	printf("=== Initial node weight===\n");
	for(i=0;i<vertices;i++)
	{
		printf("NW[%d]= %d\n ",i,node_weight_array[i]);
	}
*/

	//Initial value of min which stores the minimum node weight in each iteration of relax.
	int *min=(int*)malloc(2*sizeof(int));
	min[0]=0;
	min[1]=0;

	//GPU variable to store the minimum value
	int *gpu_min;
	checkCuda(cudaMalloc((void**)&gpu_min,2*sizeof(int)));

	while(min[0]<infinity)
	{
		min[0] = infinity;
		checkCuda(cudaMemcpy(gpu_min,min,sizeof(int),cudaMemcpyHostToDevice));

		Minimum<<<gridSize, blockSize>>>(gpu_mask_array,gpu_vertex_array,gpu_node_weight_array,gpu_edge_array,gpu_edge_weight_array,gpu_min);
		if (err != cudaSuccess) checkCuda(cudaMemcpy(node_weight_array,gpu_node_weight_array,vertex_array_size,cudaMemcpyDeviceToHost));
		{
			printf("Error: %s\n", cudaGetErrorString(err));
		}
		
		Relax<<<gridSize, blockSize>>>(gpu_mask_array,gpu_vertex_array,gpu_node_weight_array,gpu_edge_array,gpu_edge_weight_array,gpu_min);
		if (err != cudaSuccess) checkCuda(cudaMemcpy(node_weight_array,gpu_node_weight_array,vertex_array_size,cudaMemcpyDeviceToHost));
		{
			printf("Error: %s\n", cudaGetErrorString(err));
		}
		checkCuda(cudaMemcpy(node_weight_array,gpu_node_weight_array,vertex_array_size,cudaMemcpyDeviceToHost));

/*	
		//Can be uncommented to see the node weights after each iteration and/or to see the algorithm move step by step	
		printf("=== %d node weight===\n",count);
		for(i=0;i<vertices;i++)
		{
			printf("NW[%d]= %d\n ",i,node_weight_array[i]);
		}
*/			

		checkCuda(cudaMemcpy(min,gpu_min,2*sizeof(int),cudaMemcpyDeviceToHost));	
	}

	//copying the final node weights from device to host
	checkCuda(cudaMemcpy(node_weight_array,gpu_node_weight_array,vertex_array_size,cudaMemcpyDeviceToHost));

	//Stop Timer
	TIMER_END(start_time);
	printf("Kernel Execution Time: %f ms\n", start_time);

/*	
	//Can be uncommented to see the final node weights of the settled graph. i.e. the shortest distance from Source to all the nodes
	printf("=== Final node weight===\n");
	for(i=0;i<vertices;i++)
	{
		printf("NW[%d]= %d\n ",i,node_weight_array[i]);
	}
*/

	cudaFree(gpu_vertex_array);
	cudaFree(gpu_node_weight_array);
	cudaFree(gpu_edge_array);
	cudaFree(gpu_edge_weight_array);
	cudaFree(gpu_mask_array);

	free(vertex_array);
	free(node_weight_array);
	free(edge_array);
	free(edge_weight_array);
	free(mask_array);

	return 0;
}
