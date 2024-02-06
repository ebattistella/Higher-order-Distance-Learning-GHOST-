# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import csr_matrix
from networkx.algorithms.shortest_paths import all_pairs_shortest_path_length
from networkx.convert_matrix import from_scipy_sparse_array
from networkx.algorithms.approximation import max_clique
from networkx.algorithms.connectivity.connectivity import node_connectivity
import networkx.algorithms.distance_measures
import networkx.algorithms.centrality
import networkx as nx

class Graph:
    def __init__(self, matrix):
        sparse_matrix = csr_matrix(matrix)
        self.graph = from_scipy_sparse_array(sparse_matrix)
        self.max_degree = max([val for (_, val) in self.graph.degree()])
        self.paths = {}
        self.max_path = 0

    #Define a graph from a matrix of distance using a K-nearest neighbors approach
    @classmethod
    def from_data(cls, distance, k):
        closest_neigbors = np.zeros((distance.shape[0], distance.shape[0]))
        for i in range(distance.shape[0]):
            idx = np.argsort(distance[i, :])[:k]
            closest_neigbors[i, idx] = closest_neigbors[idx, i] = distance[i, idx]
        sparse_matrix = csr_matrix(closest_neigbors)
        return cls(sparse_matrix)

    #Define a graph from an adjacency matrix
    @classmethod
    def from_adjacency(cls, adjacency):
        sparse_matrix = csr_matrix(adjacency)
        return cls(sparse_matrix)


    #Add a sample to an adjacency matrix depending on its ground truth with noise in proportion proba
    #It will be used for the construction of a graph for classification in the provided synthetic example
    def add_node(self, ground, sample, proba, num_nodes):
        # If (np.random.rand() - proba >= 0) we drew a number higher than the probability so the edge isn't rnadom
        # meaning that if the ground truth of the training and test samples agree there is an edge, otherwise no edge
        # It is the opposite if (np.random.rand() - proba < 0)
        edges = np.array([int((np.random.rand() - proba >= 0) == (ground[sample] == ground[k])) for k in range(num_nodes)])
        adjacent = [node for node,edge in enumerate(edges) if edge>0]
        for k in adjacent:
            self.graph.add_edge(k, sample)
        if len(adjacent) > self.max_degree:
            self.max_degree = len(adjacent)
        self.paths[sample] = {sample:0}
        for j in range(num_nodes):
            if j in adjacent:
                self.paths[sample][j] = self.paths[j][sample] = 1
            else:
                self.paths[sample][j] = self.paths[j][sample] = min(self.max_path, 1 + min([self.paths[k][j] for k in adjacent]))

    #Compute the path length between any pair of nodes
    #In case of a disconnected graph set the infinite values to max_path (needed for the metrics computation)
    def path_finding(self, max_path):
        self.max_path = max_path
        self.paths = dict(all_pairs_shortest_path_length(self.graph, cutoff=max_path))


    #Compute the clique order metric defined as max_degree - clique_order
    #It is computed for the nodes in indexes + center
    def clique_order(self, indexes, center):
        indexes.append(center)
        aux = self.graph.subgraph(indexes)
        return self.max_degree - len(max_clique(aux))

    #Compute the eccentricity with center center in the subgraph defined by the nodes in indexes + center
    def eccentricity(self, indexes, center):
        indexes.append(center)
        subgraph = self.graph.subgraph(indexes)
        subpaths = dict({i: dict() for i in indexes})
        for i in indexes:
            for j in indexes:
                subpaths[i][j] = self.paths[i][j]
        return networkx.algorithms.distance_measures.eccentricity(subgraph, v=center, sp=subpaths)

    #Compute the connectivity in the subgraph defined by the nodes in indexes + center
    #It is defined as the number of nodes - their connectivity
    def connectivity(self, indexes, center):
        indexes.append(center)
        subgraph = self.graph.subgraph(indexes)
        return len(indexes) - node_connectivity(subgraph)
