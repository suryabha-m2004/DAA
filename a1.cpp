#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <chrono>
#include <set>
#include <cmath>
#include <string>
#include <functional>

using namespace std;

// Custom hash function for frozenset equivalent (unordered_set<int>)
struct SetHash {
    size_t operator()(const unordered_set<int>& s) const {
        size_t hash_val = 0;
        for (int x : s) {
            hash_val ^= hash<int>{}(x) + 0x9e3779b9 + (hash_val << 6) + (hash_val >> 2);
        }
        return hash_val;
    }
};

// Set equality for unordered_set
struct SetEqual {
    bool operator()(const unordered_set<int>& a, const unordered_set<int>& b) const {
        return a == b;
    }
};

// Edge structure for Dinic algorithm
struct Edge {
    int to, rev;
    double cap, flow;
};

// Dinic algorithm for max flow / min cut
class Dinic {
public:
    Dinic(int n) : n(n) {
        adj.resize(n);
        dist.resize(n);
        ptr.resize(n);
    }

    void addEdge(int u, int v, double cap) {
        adj[u].push_back({v, (int)adj[v].size(), cap, 0});
        adj[v].push_back({u, (int)adj[u].size() - 1, 0, 0});
    }

    void updateEdge(int u, int v, double newCap) {
        for(auto& edge : adj[u]) {
            if(edge.to == v) {
                edge.cap = newCap;
                return;
            }
        }
        addEdge(u, v, newCap);
    }

    void resetFlow() {
        for(int i = 0; i < n; i++) {
            for(auto& e : adj[i]) {
                e.flow = 0;
            }
        }
    }

    double maxFlow(int source, int sink) {
        double flow = 0;
        while(bfs(source, sink)) {
            fill(ptr.begin(), ptr.end(), 0);
            while(double pushed = dfs(source, sink, INF)) {
                flow += pushed;
            }
        }
        return flow;
    }

    vector<bool> getReachable(int source) {
        vector<bool> vis(n, false);
        queue<int> q;
        q.push(source);
        vis[source] = true;
        while(!q.empty()) {
            int u = q.front();
            q.pop();
            for(auto& e : adj[u]) {
                if(!vis[e.to] && (e.cap - e.flow) > 0) {
                    vis[e.to] = true;
                    q.push(e.to);
                }
            }
        }
        return vis;
    }

private:
    int n;
    vector<vector<Edge>> adj;
    vector<int> dist, ptr;
    const double INF = numeric_limits<double>::max() / 2;

    bool bfs(int source, int sink) {
        fill(dist.begin(), dist.end(), -1);
        dist[source] = 0;
        queue<int> q;
        q.push(source);
        while(!q.empty()) {
            int u = q.front();
            q.pop();
            for(auto& e : adj[u]) {
                if (e.cap - e.flow > 0 && dist[e.to] == -1) {
                    dist[e.to] = dist[u] + 1;
                    q.push(e.to);
                }
            }
        }
        return dist[sink] != -1;
    }

    double dfs(int u, int sink, double flow) {
        if(u == sink) {
            return flow;
        }
        for(int& i = ptr[u]; i < (int)adj[u].size(); i++) {
            Edge& e = adj[u][i];
            if(e.cap - e.flow > 0 && dist[e.to] == dist[u] + 1) {
                double pushed = dfs(e.to, sink, min(flow, e.cap - e.flow));
                if(pushed > 0) {
                    e.flow += pushed;
                    adj[e.to][e.rev].flow -= pushed;
                    return pushed;
                }
            }
        }
        return 0;
    }
};

// Function to find all (h-1)-cliques in the graph
vector<unordered_set<int>> enumerate_cliques(const vector<vector<int>>& graph, int h) {
    cout << "DEBUG: Starting to enumerate " << (h-1) << "-cliques" << endl;
    vector<unordered_set<int>> result;
    
    function<void(vector<int>&, int, unordered_set<int>&)> find_cliques = 
        [&](vector<int>& R, int v, unordered_set<int>& P) {
            if (R.size() == h - 1) {
                unordered_set<int> clique(R.begin(), R.end());
                result.push_back(clique);
                
                // Debug: Print the clique occasionally to see progress
                if (result.size() % 1000 == 0) {
                    cout << "DEBUG: Found " << result.size() << " cliques so far. Latest: ";
                    for (int node : R) {
                        cout << node << " ";
                    }
                    cout << endl;
                }
                return;
            }
            
            vector<int> P_copy(P.begin(), P.end());
            for (auto u : P_copy) {
                if (u <= v) continue;  // Only consider vertices with higher indices
                
                // Add u to R
                R.push_back(u);
                P.erase(u);
                
                // Find neighbors of u that are in P
                unordered_set<int> new_P;
                for (auto w : graph[u]) {
                    if (P.find(w) != P.end()) {
                        new_P.insert(w);
                    }
                }
                
                // Recursive call
                find_cliques(R, u, new_P);
                
                // Remove u from R
                R.pop_back();
                P.insert(u);
            }
        };
    
    for (int i = 0; i < graph.size(); i++) {
        cout << "DEBUG: Starting clique search from vertex " << i << "/" << graph.size() << endl;
        vector<int> R = {i};
        unordered_set<int> P;
        for (int neighbor : graph[i]) {
            if (neighbor > i) {  // Only consider vertices with higher indices
                P.insert(neighbor);
            }
        }
        
        cout << "DEBUG: Vertex " << i << " has " << P.size() << " eligible neighbors" << endl;
        find_cliques(R, i, P);
    }
    
    cout << "DEBUG: Enumeration complete. Found " << result.size() << " cliques of size " << (h-1) << endl;
    return result;
}

int main() {
    int h;
    cout << "Enter the value of h: ";
    cin >> h;
    cout << "DEBUG: Using h = " << h << endl;
    
    ifstream file("test1.txt");
    if (!file.is_open()) {
        cerr << "Failed to open the file." << endl;
        return 1;
    }
    
    int n_max, m;
    file >> n_max >> m;
    cout << "DEBUG: File reports " << n_max << " max node ID and " << m << " edges" << endl;
    
    // Read all edges and find unique vertices
    vector<pair<int, int>> edges;
    set<int> unique_vertices;
    
    for (int i = 0; i < m; ++i) {
        int u, v;
        file >> u >> v;
        if(u == v) {
            cout << "DEBUG: Skipping self-loop for vertex " << u << endl;
            continue; // Skip self-loops
        }
        if(count(edges.begin(), edges.end(), make_pair(u, v)) > 0) {
            cout << "DEBUG: Skipping duplicate edge " << u << "-" << v << endl;
            continue; // Skip duplicate edges
        }
        if(count(edges.begin(), edges.end(), make_pair(v, u)) > 0) {
            cout << "DEBUG: Skipping reverse duplicate edge " << u << "-" << v << endl;
            continue; // Skip reverse duplicate edges
        }
        edges.push_back({u, v});
        unique_vertices.insert(u);
        unique_vertices.insert(v);
    }
    
    cout << "DEBUG: After filtering, have " << edges.size() << " edges and " << unique_vertices.size() << " unique vertices" << endl;
    
    // Create a mapping from original vertex IDs to new IDs (0 to n-1)
    unordered_map<int, int> orig_to_new;
    unordered_map<int, int> new_to_orig;
    int idx = 0;
    
    for (int v : unique_vertices) {
        orig_to_new[v] = idx;
        new_to_orig[idx] = v;
        idx++;
    }
    
    int n = unique_vertices.size(); // Actual number of nodes
    cout << "DEBUG: Renumbered vertices from 0 to " << (n-1) << endl;
    
    // Create the renumbered graph
    vector<vector<int>> G(n);
    for (const auto& edge : edges) {
        int u_new = orig_to_new[edge.first];
        int v_new = orig_to_new[edge.second];
        G[u_new].push_back(v_new);
        G[v_new].push_back(u_new);  // Assuming undirected graph
    }
    
    cout << "DEBUG: Created adjacency list representation of graph" << endl;
    
    // Print degree stats
    vector<int> degrees(n, 0);
    for (int i = 0; i < n; i++) {
        degrees[i] = G[i].size();
    }
    sort(degrees.begin(), degrees.end());
    cout << "DEBUG: Degree stats - Min: " << degrees[0] 
         << ", Median: " << degrees[n/2] 
         << ", Max: " << degrees[n-1] << endl;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Find all (h-1)-cliques
    vector<unordered_set<int>> Lambda = enumerate_cliques(G, h);
    
    // Compute deg_Ψ for each vertex
    cout << "DEBUG: Computing clique membership counts (deg_Psi) for vertices" << endl;
    vector<int> deg_Psi(n, 0);
    for (const auto& sigma : Lambda) {
        if (sigma.empty()) {
            cout << "DEBUG: Warning - empty clique detected" << endl;
            continue;
        }
        
        // Find common neighbors
        unordered_set<int> common_neighbors;
        bool first = true;
        
        for (int v : sigma) {
            if (first) {
                for (int neighbor : G[v]) {
                    if (sigma.find(neighbor) == sigma.end()) {
                        common_neighbors.insert(neighbor);
                    }
                }
                first = false;
            } else {
                unordered_set<int> new_common;
                for (int neighbor : G[v]) {
                    if (sigma.find(neighbor) == sigma.end() && 
                        common_neighbors.find(neighbor) != common_neighbors.end()) {
                        new_common.insert(neighbor);
                    }
                }
                common_neighbors = move(new_common);
            }
        }
        
        for (int v : common_neighbors) {
            deg_Psi[v]++;
        }
    }
    
    // Find maximum degree
    int max_deg = 0;
    for (int d : deg_Psi) {
        max_deg = max(max_deg, d);
    }
    
    cout << "DEBUG: Maximum deg_Psi: " << max_deg << endl;
    
    // Count non-zero deg_Psi
    int non_zero_deg = 0;
    for (int d : deg_Psi) {
        if (d > 0) non_zero_deg++;
    }
    cout << "DEBUG: " << non_zero_deg << " vertices have non-zero deg_Psi out of " << n << " total" << endl;
    
    double l = 0.0, u = static_cast<double>(max_deg);
    double epsilon = (n > 1) ? 1.0 / (n * (n - 1)) : 1e-9;
    cout << "DEBUG: Search range initialized to [" << l << ", " << u << "] with epsilon = " << epsilon << endl;
    
    vector<int> D;
    int iteration = 0;
    
    while (u - l > epsilon) {
        iteration++;
        double lambda = (l + u) / 2.0;
        cout << "Iteration " << iteration << ": λ = " << lambda 
             << ", Search range = [" << l << ", " << u << "]" << endl;
        
        // Create flow network
        int num_nodes = 2 + n + Lambda.size();  // s, t, vertices, cliques
        cout << "DEBUG: Creating flow network with " << num_nodes << " nodes (2 + " 
             << n << " vertices + " << Lambda.size() << " cliques)" << endl;
        
        int s = 0;  // source
        int t = 1;  // sink
        int vertex_offset = 2;
        int clique_offset = vertex_offset + n;
        
        // Initialize Dinic algorithm with the number of nodes
        Dinic dinic(num_nodes);
        
        // Add edges from source to vertices
        int source_edges = 0;
        for (int v = 0; v < n; ++v) {
            if (deg_Psi[v] > 0) {
                dinic.addEdge(s, vertex_offset + v, deg_Psi[v]);
                source_edges++;
            }
        }
        cout << "DEBUG: Added " << source_edges << " edges from source to vertices" << endl;
        
        // Add edges from vertices to sink
        for (int v = 0; v < n; ++v) {
            dinic.addEdge(vertex_offset + v, t, h * lambda);
        }
        cout << "DEBUG: Added " << n << " edges from vertices to sink with capacity " << (h * lambda) << endl;
        
        // Add edges from cliques to vertices and vertices to cliques
        int clique_vertex_edges = 0;
        int vertex_clique_edges = 0;
        
        for (size_t i = 0; i < Lambda.size(); ++i) {
            const auto& sigma = Lambda[i];
            int clique_node = clique_offset + i;
            
            for (int v : sigma) {
                dinic.addEdge(clique_node, vertex_offset + v, numeric_limits<double>::infinity());
                clique_vertex_edges++;
            }
            
            // Find common neighbors
            unordered_set<int> common_neighbors;
            bool first = true;
            
            for (int v : sigma) {
                if (first) {
                    for (int neighbor : G[v]) {
                        if (sigma.find(neighbor) == sigma.end()) {
                            common_neighbors.insert(neighbor);
                        }
                    }
                    first = false;
                } else {
                    unordered_set<int> new_common;
                    for (int neighbor : G[v]) {
                        if (sigma.find(neighbor) == sigma.end() && 
                            common_neighbors.find(neighbor) != common_neighbors.end()) {
                            new_common.insert(neighbor);
                        }
                    }
                    common_neighbors = move(new_common);
                }
            }
            
            for (int v : common_neighbors) {
                dinic.addEdge(vertex_offset + v, clique_node, 1.0);
                vertex_clique_edges++;
            }
            
            // Debugging: Occasionally print progress
            if (i % 1000 == 0 && i > 0) {
                cout << "DEBUG: Processed " << i << "/" << Lambda.size() << " cliques for flow network" << endl;
            }
        }
        
        cout << "DEBUG: Added " << clique_vertex_edges << " edges from cliques to vertices" << endl;
        cout << "DEBUG: Added " << vertex_clique_edges << " edges from vertices to cliques" << endl;
        
        // Compute min cut using Dinic algorithm
        cout << "DEBUG: Computing min cut with Dinic algorithm" << endl;
        double cut_value = dinic.maxFlow(s, t);
        cout << "DEBUG: Min cut value: " << cut_value << endl;
        
        // Get the reachable vertices from source after max flow
        vector<bool> min_cut_set = dinic.getReachable(s);
        
        bool only_source_in_S = true;
        for (int i = 1; i < num_nodes; ++i) {
            if (min_cut_set[i]) {
                only_source_in_S = false;
                break;
            }
        }
        
        if (only_source_in_S) {
            cout << "DEBUG: Only source in cut set, updating upper bound" << endl;
            u = lambda;
        } else {
            cout << "DEBUG: Found non-trivial cut, updating lower bound" << endl;
            l = lambda;
            D.clear();
            for (int v = 0; v < n; ++v) {
                if (min_cut_set[vertex_offset + v]) {
                    D.push_back(v);
                }
            }
            cout << "DEBUG: Current solution has " << D.size() << " vertices" << endl;
        }
        
        if (abs(u - l) < 1e-12) {
            cout << "Early stopping: negligible change in bounds" << endl;
            break;
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(end_time - start_time).count();
    
    cout << "DEBUG: Binary search complete after " << iteration << " iterations in " << elapsed_time << " seconds" << endl;
    
    // Print results
    if (!D.empty()) {
        // Compute subgraph properties
        int num_vertices = D.size();
        int num_edges = 0;
        int self_loops = 0;
        
        for (size_t i = 0; i < D.size(); ++i) {
            for (size_t j = i; j < D.size(); ++j) {
                int u = D[i];
                int v = D[j];
                
                // Check if edge exists
                bool edge_exists = false;
                for (int neighbor : G[u]) {
                    if (neighbor == v) {
                        edge_exists = true;
                        break;
                    }
                }
                
                if (edge_exists) {
                    num_edges++;
                    if (u == v) {
                        self_loops++;
                    }
                }
            }
        }
        
        cout << "\nFinal Results:" << endl;
        cout << "Densest Subgraph Vertices (Original IDs): ";
        
        // Convert back to original IDs for output
        vector<int> original_D;
        for (int v : D) {
            original_D.push_back(new_to_orig[v]);
        }
        sort(original_D.begin(), original_D.end());
        
        cout << "DEBUG: Converting " << D.size() << " vertices back to original IDs" << endl;
        
        for (size_t i = 0; i < original_D.size(); ++i) {
            cout << original_D[i];
            if (i < original_D.size() - 1) cout << ", ";
        }
        cout << endl;
        
        cout << (h-1) << "-clique Density: " << round(l * 10000) / 10000 << endl;
        cout << "Number of vertices: " << num_vertices << endl;
        cout << "Number of edges: " << num_edges << " (including " << self_loops << " self-loops)" << endl;
        cout << "Total Iterations: " << iteration << endl;
        cout << "Execution Time: " << elapsed_time << " seconds" << endl;
    } else {
        cout << "\nNo densest subgraph found." << endl;
        cout << "Total Iterations: " << iteration << endl;
        cout << "Execution Time: " << elapsed_time << " seconds" << endl;
    }
    
    return 0;
}
