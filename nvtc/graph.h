#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <utility>
#include <map>

typedef std::vector< std::pair<int, int> > Edges;
typedef std::vector< std::vector<int> > AdjList;

int NumVertices(const Edges& edges);

void ReadEdgesFromFile(const char* filename, Edges& edges);
void WriteEdgesToFile(const Edges& edges, const char* filename);

void RemoveDuplicateEdges(Edges* edges);
void RemoveSelfLoops(Edges* edges);
void MakeUndirected(Edges* edges);
void PermuteEdges(Edges* edges);
void PermuteVertices(Edges* edges);

inline void NormalizeEdges(Edges* edges) {
  MakeUndirected(edges);
  RemoveDuplicateEdges(edges);
  RemoveSelfLoops(edges);
  PermuteEdges(edges);
  PermuteVertices(edges);
}

AdjList EdgesToAdjList(const Edges& edges);

std::pair<int, uint64_t> read_binfile_to_arclist_v2(const char* file_name, std::vector<std::pair<int, int>>& arcs);
#endif
