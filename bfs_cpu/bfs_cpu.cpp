#include<iostream>
#include <list>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

// class of a graph with methods
class BFSGraph {
    int vertex;
    list<int> *adjacency;
public:
    BFSGraph(int vertex);
    void addEdge(int v, int w);
    void BFS(int startingPoint);
};

// BFS traversal method with a starting point node
void BFSGraph::BFS(int startingPoint) {

    bool *visitedVertex = new bool[vertex];
    list<int> Graphqueue;
    Graphqueue.push_back(startingPoint);
    list<int>::iterator ite;    

    // mark all node visited as false
    for(int i = 0; i < vertex; i++) {
        visitedVertex[i] = false;
    }

    // mark start index nodes as true
    visitedVertex[startingPoint] = true;
    
    while(!Graphqueue.empty()) {
        startingPoint = Graphqueue.front();
        Graphqueue.pop_front();

	// mark visited node as true
        for (ite = adjacency[startingPoint].begin(); ite != adjacency[startingPoint].end(); ++ite) {
            if (visitedVertex[*ite] == false) {
                visitedVertex[*ite] = true;
                Graphqueue.push_back(*ite);
            }
        }
    }
}

// Assigning values to vertex and fill list
BFSGraph::BFSGraph(int vertex) {
    this->vertex = vertex;
    adjacency = new list<int>[vertex];
}

// add edges in the graph
void BFSGraph::addEdge(int v, int w) {
    adjacency[v].push_back(w);
}

// main method
int main(int argc, char* argv[]) {

    // User input
    int numberOfNodes = atoi(argv[1]);
    BFSGraph g(numberOfNodes*2);

    // Time computation   
    clock_t t1;
    t1 = clock();

    // populate the graph
    for(int i=0;i<numberOfNodes;i++) {
	g.addEdge(i, rand()%numberOfNodes);
	g.addEdge(i, rand()%numberOfNodes);
    }

    g.BFS(0);
    
    t1 = clock() - t1; 
    double time_taken = ((double)t1)/CLOCKS_PER_SEC; // in seconds 
 
    return 0;
}

