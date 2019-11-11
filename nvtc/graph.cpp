#include "graph.h"

#include <algorithm>
#include <fstream>
#if __GNUG__
#include <bits/stdc++.h>
#else
#define INT_MAX 2147483647
#endif

using namespace std;

unsigned long get_edge(std::ifstream& fin){
    fin.seekg(0, fin.end);
    unsigned long edge_size = fin.tellg();
    fin.seekg(0, fin.beg);    
    if (edge_size % 8 != 0){
        throw std::logic_error( std::string{} + "not multiply of 8 at " +  __FILE__ +  ":" + std::to_string(__LINE__));
    }
    return edge_size / 8;
}

//! V2 allows node with zero degree
std::pair<int, int> read_binfile_to_arclist_v2(const char* file_name, std::vector<std::pair<int, int>>& arcs){
    std::ifstream fin;
    fin.open(file_name, std::ifstream::binary | std::ifstream::in);
    unsigned long file_size = get_edge(fin);
#if VERBOSE
    std::cout << "num of edges before cleanup: " << file_size << std::endl;
#endif
    arcs.resize(file_size);
    fin.read((char*)arcs.data(), 2 * file_size * sizeof(int));
    int node_num = 0;
    for(std::vector<std::pair<int, int>>::iterator it = arcs.begin(); it != arcs.end(); ++it){
        if(it->first > node_num) {
            node_num = it->first;
        }
        else if(it->second > node_num) {
            node_num = it->second;
        }
        if(it->first == it->second) {
            it->first = INT_MAX;
            it->second = INT_MAX;
        }
        else if(it->first > it->second) {
            swap(it->first, it->second);
        }
        
    }
    // sort arcs
    std::sort(arcs.begin(), arcs.end()); 
    // remove the duplicate
    std::pair<int, int> last_value = arcs[0];
    for(unsigned long i = 1; i < arcs.size() - 1; i++){
        while(arcs[i].first == last_value.first && arcs[i].second == last_value.second){
            arcs[i].first = INT_MAX;
            arcs[i].second = INT_MAX;
            i++;
        }
        last_value = arcs[i];
    }
    // sort arcs again
    std::sort(arcs.begin(), arcs.end());
    // find the number of duplicate edges
    int edges = 0;
    while(edges < arcs.size()){
        if(arcs[edges].first == INT_MAX){
            break;
        }
        edges++;
    }
    arcs.resize(edges);
    return std::make_pair(node_num + 1, edges);
}

std::pair<int, int> read_binfile_to_arclist(const char* file_name, std::vector<std::pair<int, int>>& arcs){
    std::ifstream fin;
    fin.open(file_name, std::ifstream::binary | std::ifstream::in);
    unsigned long file_size = get_edge(fin);
#if VERBOSE
#if TIMECOUNTING
    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
#endif
    std::cout << "Start file reading..." << std::endl;
    int base_counter = file_size / 10 + 1;
#endif
    char u_array[4], v_array[4];
    unsigned int *u, *v;
    std::map<int, int> kv_map; 
    std::map<std::pair<int,int>, bool> arc_exist_map;
    int node_id = 1;
    for(unsigned long i = 0; i < file_size; i++){
#if VERBOSE
    if(i % base_counter == 1)
        std::cout << 10 * i / base_counter << "% processed for input file"  << std::endl;    
#endif        
        fin.read(u_array, 4);
        fin.read(v_array, 4);
        u = (unsigned int*)u_array;
        v = (unsigned int*)v_array;
        int& u_id = kv_map[*u];
        if(u_id == 0){
            u_id = node_id;
            node_id ++;
        }
        int& v_id = kv_map[*v];
        if(v_id == 0){
            v_id = node_id;
            node_id ++;
        }
        if(u_id < v_id){
            bool& arc_exist = arc_exist_map[std::make_pair(u_id, v_id)];
            if(arc_exist)
                continue;
            arc_exist = true;
        }
        else if(u_id > v_id){
            bool& arc_exist = arc_exist_map[std::make_pair(v_id, u_id)];
            if(arc_exist)
                continue;
            arc_exist = true;            
        }       
    }
    int actual_edge_num = arc_exist_map.size();
#if VERBOSE
#if TIMECOUNTING
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    std::chrono::system_clock::duration dtn = end_time - start_time;
    float time_used = std::chrono::duration_cast<std::chrono::milliseconds>(dtn).count()/1000.0;
    std::cout << "File reading finished, Time used: " << time_used << "s" << std::endl;
#else
    std::cout << "File reading finished" << std::endl;
#endif     
    std::cout << "Actual node size " << node_id - 1<< std::endl;
    std::cout << "Actual edges " << actual_edge_num << std::endl;
#endif    
    fin.close();
    arcs.reserve(actual_edge_num);
    for(std::map<std::pair<int,int>, bool>::iterator it = arc_exist_map.begin(); it != arc_exist_map.end(); ++it){
        arcs.push_back(std::make_pair(it->first.first -1, it->first.second - 1));
    }
    return std::make_pair(node_id - 1, actual_edge_num);
}

void ReadEdgesFromFile(const char* filename, Edges& edges) {
  read_binfile_to_arclist(filename, edges);
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
