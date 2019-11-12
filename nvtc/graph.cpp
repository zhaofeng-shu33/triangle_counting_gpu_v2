#include "graph.h"

#include <algorithm>
#include <fstream>
#if __GNUG__
#include <bits/stdc++.h>
#else
#define INT_MAX 2147483647
#endif

using namespace std;

uint64_t get_edge(std::ifstream& fin){
    fin.seekg(0, fin.end);
    uint64_t edge_size = fin.tellg();
    fin.seekg(0, fin.beg);    
    if (edge_size % 8 != 0){
        throw std::logic_error( std::string{} + "not multiply of 8 at " +  __FILE__ +  ":" + std::to_string(__LINE__));
    }
    return edge_size / 8;
}

//! V2 allows node with zero degree
std::pair<int, uint64_t> read_binfile_to_arclist_v2(const char* file_name, int*& arcs){
    std::ifstream fin;
    fin.open(file_name, std::ifstream::binary | std::ifstream::in);
    uint64_t file_size = get_edge(fin);
#if VERBOSE
    std::cout << "num of edges before cleanup: " << file_size << std::endl;
#endif
    arcs = (int*)malloc(2 * sizeof(int) * file_size);
    fin.read(reinterpret_cast<char*>(arcs),
        2 * file_size * sizeof(int));
    int node_num = 0;
    for (int i = 0;
        i < file_size; ++i) {
        if (arcs[2 * i] > node_num) {
            node_num = arcs[2 * i];
        } else if (arcs[2 * i + 1] > node_num) {
            node_num = arcs[2 * i + 1];
        }
        if (arcs[2 * i + 1] == arcs[2 * i]) {
            arcs[2 * i + 1] = INT_MAX;
            arcs[2 * i] = INT_MAX;
        } else if (arcs[2 * i] > arcs[2 * i + 1]) {
            std::swap(arcs[2 * i], arcs[2 * i + 1]);
        }
    }
    // sort arcs
    uint64_t* arcs_start_ptr = (uint64_t*)arcs;
    std::sort(arcs_start_ptr, arcs_start_ptr + file_size);
    // remove the duplicate
    uint64_t* last_value = (uint64_t*)arcs;
    uint64_t j = 1;
    for (uint64_t i = 1; i < file_size - 1; i++) {
        while (*(last_value + j - 1) == *(last_value + i)) {
            arcs[2 * i] = INT_MAX;
            arcs[2 * i + 1] = INT_MAX;
            i++;
        }
        j = i + 1;
    }
    // sort arcs again
    std::sort(arcs_start_ptr, arcs_start_ptr + file_size);
    // find the number of duplicate edges
    uint64_t edges = 0;
    while (edges < file_size) {
        if (arcs[2 * edges] == INT_MAX) {
            break;
        }
        edges++;
    }
    return std::make_pair(node_num + 1, edges);
}


void WriteEdgesToFile(const Edges& edges, const char* filename) {
  ofstream out(filename, ios::binary);
  int m = edges.size();
  out.write((char*)&m, sizeof(int));
  out.write((char*)edges.data(), 2 * m * sizeof(int));
}

int NumVertices(const Edges& edges) {
  int num_vertices = 0;
  for (const pair<int, int>& edge : edges)
    num_vertices = max(num_vertices, 1 + max(edge.first, edge.second));
  return num_vertices;
}

void RemoveDuplicateEdges(Edges* edges) {
  sort(edges->begin(), edges->end());
  edges->erase(unique(edges->begin(), edges->end()), edges->end());
}

void RemoveSelfLoops(Edges* edges) {
  for (int i = 0; i < edges->size(); ++i) {
    if ((*edges)[i].first == (*edges)[i].second) {
      edges->at(i) = edges->back();
      edges->pop_back();
      --i;
    }
  }
}

void MakeUndirected(Edges* edges) {
  const size_t n = edges->size();
  for (int i = 0; i < n; ++i) {
    pair<int, int> edge = (*edges)[i];
    swap(edge.first, edge.second);
    edges->push_back(edge);
  }  
}

void PermuteEdges(Edges* edges) {
  random_shuffle(edges->begin(), edges->end());
}

void PermuteVertices(Edges* edges) {
  vector<int> p(NumVertices(*edges));
  for (int i = 0; i < p.size(); ++i)
    p[i] = i;
  random_shuffle(p.begin(), p.end());
  for (pair<int, int>& edge : *edges) {
    edge.first = p[edge.first];
    edge.second = p[edge.second];
  }
}

AdjList EdgesToAdjList(const Edges& edges) {
  // Sorting edges with std::sort to optimize memory access pattern when
  // creating graph gives less than 20% speedup.
  AdjList graph(NumVertices(edges));
  for (const pair<int, int>& edge : edges)
    graph[edge.first].push_back(edge.second);
  return graph;
}
