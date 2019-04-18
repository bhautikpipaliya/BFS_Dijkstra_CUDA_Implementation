#ifndef GRAPH_H
#define GRAPH_H
#include <map>

class Graph
{
	struct Edge
	{
		int v;
		int weight;
		Edge *next;
	};


	struct Vertex
	{

		// Used in all algorithms, either to record a sub-graph forming
		// a tree (breadth-first or depth-first tree) or to record
		// shortest paths.
		int parent;


		// Used in Dijkstra(). This are references to the positions in a
		// multimap (binary search tree) where a vertex has been stored.
		std::multimap<int, int>::iterator it;

		//Use to measure node distance in Dijkistra
		int distance;

		// Adjacency list
		Edge *edges;
	};


	// Number of vertices
	int size;
	
	// Array of vertices
	Vertex *V;

	// Auxiliary function for shortest paths
	void Relax(int u, Edge *edge);

	public:
		// Constructor
		Graph(int size);
	
		// Destructor	
		~Graph();

		// Return the number of vertices
		int getSize() { return size; }

		// Adds a new undirected (or bidirectional) edge. This consists in
		// adding one edge u->v and another edge v->u. This function is defined
		// inline.
		void AddUndirectedEdge(int u, int v, int weight);

		// Print the graph
		void Print();

		// Dijkstra algorithm to calculate single-source shortest paths on a
		// weighted, directed graph with non-negative weights.
		void Dijkstra(int s);
};

#endif
