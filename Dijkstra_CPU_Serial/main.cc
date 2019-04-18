#include <iostream>
#include "Graph.h"
#include <string.h>
#include <stdlib.h> 
#include <time.h>

#define MAX_WEIGHT 10

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}



int main(int argc,char* argv[])
{

	//Number of vertices
	int n = atoi(argv[1]);

	int degree = atoi(argv[2]);

	// Undirected graph shown in this document for the trace of the BFS algorithm.
	Graph g(n);

	int vertex[n];
	int vertex_copy[n];
	
	//Time variables
	double start,stop;

	size_t vertex_size = n*sizeof(int);

	//Iterative variable
	int i,j,k;

	for(i=0;i<n;i++)
	{
		vertex[i]=i;
	}

	memcpy(vertex_copy,vertex,vertex_size);
	
	//Variable to store edge weight
	int edge_weight;

	//Setting seed of the RNG to system clock
	srand(time(NULL));

	//Temp variable
	int temp;

	for(i=0;i<n;i++)
	{
                //Function to jumble the nodes in the vertex array and assign them to each node
                for(j=n-1;j>0;j--)
                {
                        k=rand()%(j+1);
                        temp = vertex_copy[j];
                        vertex_copy[j]=vertex_copy[k];
                        vertex_copy[k]=temp;
                }

		//Initializing the graph with random edges based on the input degree function
                for(j=0;j<degree;j++)
                {
                        if(vertex_copy[j]==i)
                        {
                                j=j+1;
				edge_weight=rand()%MAX_WEIGHT+1;	
				g.AddUndirectedEdge(i,vertex_copy[j],edge_weight);
   			}                            
                        else
			{
				edge_weight=rand()%MAX_WEIGHT+1;	
				g.AddUndirectedEdge(i,vertex_copy[j],edge_weight);
			}
		}	


	}

	start=CLOCK();
	g.Dijkstra(0);	
	stop = CLOCK();
	double final = (stop-start)/1000;
	std::cout<< "Serial Execution Time: "<< final << " ms\n"; 


}
