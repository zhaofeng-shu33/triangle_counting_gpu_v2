#include "counting_cpu.h"

uint64_t CpuForward(const Edges& edges, int node_num) {
   int m = edges.size();
   int* dev_edges = new int [ 2 * m ];
   int n = node_num;
   // Unzip Edges
   for (int i = 0; i < m; i++) {
        dev_edges[i] = edges[i].first;
        dev_edges[m+i] = edges[i].second;
   }
   int* dev_nodes = new int [n + 1];
   // Calculate NodePointers
   for (int i = 0; i <= m; i++) {
      int prev = i > 0 ? dev_edges[m + i - 1] : -1;
      int next = i < m ? dev_edges[m + i] : n;
      for (int j = prev + 1; j <= next; j++)
        dev_nodes[j] = i;  
   }
   // Calculate Triangles
   uint64_t count = 0;
   for (int i = 0; i < m; i++) {
     int u = dev_edges[i], v = dev_edges[m + i];
     int u_it = dev_nodes[u], u_end = dev_nodes[u + 1];
     int v_it = dev_nodes[v], v_end = dev_nodes[v + 1];
     int a = dev_edges[u_it], b = dev_edges[v_it];
     while (u_it < u_end && v_it < v_end) {
       int d = a - b;
       if (d <= 0)
         a = dev_edges[++u_it];
       if (d >= 0)
         b = dev_edges[++v_it];
       if (d == 0)
         ++count;
     }       
   }
   delete dev_nodes;
   delete dev_edges;
   return count;
}

