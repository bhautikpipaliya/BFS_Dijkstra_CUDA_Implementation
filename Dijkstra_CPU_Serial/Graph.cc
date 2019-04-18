#include <climits>
#include <iostream>
#include <queue>
#include "Graph.h"

Graph::Graph(int size)
{
	this->size = size;
	V = new Vertex[size]();
	for (int i = 0; i < size; i++)
	{
		V[i].parent = -1;
		V[i].edges = NULL;
		V[i].distance = INT_MAX;	
	}
}

Graph::~Graph()
{
	for (int i = 0; i < size; i++)
	{
		while (V[i].edges)
		{
			Edge *edge = V[i].edges;
			V[i].edges = edge->next;
			delete edge;
		}
	}

	delete[] V;
}

void Graph::AddUndirectedEdge(int u, int v, int weight)
{
	Edge *edge = new Edge();
	edge->v = v;
	edge->weight = weight;
	edge->next = V[u].edges;
	V[u].edges = edge;
}

void Graph::Print()
{
	std::cout << "Printing graph:\n";
	for (int i = 0; i < size; i++)
	{
		std::cout << "Vertex " << i << ":";
		if (V[i].parent >= 0)
			std::cout << " parent=" << V[i].parent;

		std::cout << ", edges={";
		
		for (Edge *edge = V[i].edges; edge; edge = edge->next)
			std::cout << ' ' << i << "->" << edge->v;
		std::cout << " }\n";
	}
}


void Graph::Relax(int u, Edge *edge)
{
	int v = edge->v;
	if (V[u].distance + edge->weight < V[v].distance && V[u].distance != INT_MAX)
	{
		V[v].parent = u;
		V[v].distance = V[u].distance + edge->weight;
	}
}

void Graph::Dijkstra(int s)
{
	// Initialize
	for (int i = 0; i < size; i++)
	{
		V[i].distance = INT_MAX;
		V[i].parent = -1;
	}

	V[s].distance = 0;

	// Insert all vertexes in a binary search tree, where the key is the
	// current distance of a vertex (field 'distance'), and the data is the
	// vertex index. Each vertex stores its position in the binary search
	// tree using an iterator (field 'it').
	std::multimap<int, int> tree;
	for (int i = 0; i < size; i++)
	V[i].it = tree.insert(std::pair<int, int>(V[i].distance, i));
	
	// Iterate until tree is empty
	while (tree.size())
	{
		// Get minimum element in tree
		std::multimap<int, int>::iterator it = tree.begin();
		int u = it->second;
		// Remove element from the tree and set its associated iterator
		// to a past-the-end iterator.
		tree.erase(it);
		V[u].it = tree.end();
		
		// Traverse edges
		for (Edge *edge = V[u].edges; edge; edge = edge->next)
		{
			// Obtain destination vertex
			int v = edge->v;

			// Check if vertex is in the tree
			bool is_in_tree = V[v].it != tree.end();

			// Extract v from tree to update its key (distance)
			if (is_in_tree)
			{
				tree.erase(V[v].it);
			}

			// Relax edge
			Relax(u, edge);

			// Insert v again with its new key (distance)
			if (is_in_tree)
			{
				V[v].it = tree.insert(std::pair<int, int>(V[v].distance, v));
			}
		}
	}
}
