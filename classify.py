#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import os
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import metrics
import graph
import multiprocessing as mp
from itertools import combinations_with_replacement
from scipy.special import binom
from scipy.stats import kendalltau, pearsonr, spearmanr
from time import time


def parallel_dist_order(g, method, combination, center, w):
    methods = [graph.Graph.clique_order, graph.Graph.eccentricity, graph.Graph.connectivity]

    if w != 0:
        return w * methods[method](g, list(combination), center)
    else:
        return 0


# compute the higher order distance to centers
def to_centers_distance_matrix(data_train, data_test, w, centers, clusters, order,
                               path, graph_train=[], cluster_metric=1, second_order_dist=[]):
    order += 1
    methods_func = [graph.Graph.clique_order, graph.Graph.eccentricity, graph.Graph.connectivity]

    distance_matrix = np.zeros((np.shape(data_test)[0], len(centers)))

    num_train_nodes = len(graph_train[0].graph.nodes())
    for patient in range(data_test.shape[0]):
        if patient >= num_train_nodes:
            #Example when we do have a graoph structure, here it is meant to be used with the synthetic test set
            g = copy.deepcopy(graph_train[0])
            g.add_node(graph_train[1], patient, graph_train[2], num_train_nodes)
        else:
            g = graph_train[0]
        for idx_train in range(len(centers)):
            if cluster_metric != 0:
                cluster_k = [i for i in range(len(clusters)) if clusters[i] == clusters[centers[idx_train]]]
            # We add the second order metrics
            distance_matrix[patient, idx_train] += np.sum([
                w[i] *
                second_order_dist[i](data_test[patient, i], data_train[centers[idx_train], i]) for i in
                range(len(second_order_dist)) if w[i] != 0])

            if w[-1] != 0:
                aux_para = 0
                # Higher-order components
                if cluster_metric > 0:
                    aux_mult = w[-1] / len(cluster_k) ** 2
                    if not os.path.exists(path + "p" + str(patient) + "center" + str(idx_train) + "graph" + ".npy"):
                        for p2 in cluster_k:
                            p = centers[idx_train]
                            if g.paths[p][p2] != 0:
                                aux_para += (g.paths[patient][p] + g.paths[p2][patient]) / g.paths[p][p2]
                        np.save(path + "p" + str(patient) + "center" + str(idx_train) + "graph", aux_para)
                    else:
                        aux_para = np.load(path + "p" + str(patient) + "center" + str(idx_train) + "graph" + ".npy",
                                           allow_pickle=True)
                    aux_para = aux_mult * aux_para
                # Cluster-wise component
                if cluster_metric == -1 or cluster_metric == 2:
                    aux_para += w[-1]*sum(
                        [g.paths[patient][q] + sum([g.paths[p2][q] for p2 in cluster_k if p2 > q]) for q in cluster_k
                         ]) / np.sum([g.paths[q][p2]
                                      for q in cluster_k for p2 in cluster_k if p2 > q])
                distance_matrix[patient, idx_train] += aux_para
            pool = mp.Pool(processes=10)
            for method_i in range(len(methods_func)):
                if w[-1 - 1 - method_i] != 0:
                    aux_para = 0
                    if cluster_metric > 0:
                        if not os.path.exists(path + "p" + str(patient) + "center" + str(idx_train) + "method" + str(
                                method_i) + ".npy"):
                            aux_para += sum(pool.starmap(parallel_dist_order,
                                                         ((g, method_i, list(combination) + [idx_train],
                                                           patient,
                                                           1 / binom(len(cluster_k) - 2 + current_order - 1 - 2,
                                                                     current_order - 2))
                                                          for current_order in range(2, order) for combination in
                                                          combinations_with_replacement(cluster_k, current_order - 2))))
                            np.save(path + "p" + str(patient) + "center" + str(idx_train) + "method" + str(method_i),
                                    aux_para)
                        else:
                            aux_para += np.load(path + "p" + str(patient) + "center" + str(idx_train) + "method" + str(
                                method_i) + ".npy",
                                                allow_pickle=True)
                    if cluster_metric == -1 or cluster_metric == 2:
                        aux_para += parallel_dist_order(g, method_i, cluster_k, patient, 1)
                    distance_matrix[patient, idx_train] += w[-1 - 1 - method_i] * aux_para
            pool.close()
    return distance_matrix


def aggregate_dist(dist, cluster_index, pred_index, medoid, linkage):
    if linkage == "Average":
        return np.mean(dist[pred_index, cluster_index])
    elif linkage == "Min":
        return np.min(dist[pred_index, cluster_index])
    elif linkage == "Min Max":
        return np.max(dist[pred_index, cluster_index])
    elif linkage == "Min center":
        return np.min(dist[pred_index, medoid])
    print("Undefined Linkage")
    exit(1)


def neigbhor_predict(dist_to_centers, simi):
    print("dist", dist_to_centers, np.shape(dist_to_centers))
    ope = np.argmin
    if simi:
        ope = np.argmax
    predictions = ope(dist_to_centers, axis=1)
    print(predictions, len(predictions))
    return predictions


def KNN(cluster, dist_to_pred, k):
    predictions = []
    for i in range(np.shape(dist_to_pred)[0]):
        closest = np.argsort(dist_to_pred[i, :])[:k]
        aux_classes = np.unique([cluster[j] for j in closest])
        aux_votes = [len([1 for j in closest if cluster[j] == class_i]) for class_i in aux_classes]
        predictions.append(aux_classes[np.argmax(aux_votes)])
    return predictions


def assess(clustering, y_pred):
    acc = metrics.balanced_accuracy_score(clustering, y_pred)
    prec = metrics.precision_score(clustering, y_pred, average='weighted')
    rec = metrics.recall_score(clustering, y_pred, average='weighted')
    spec = (np.sum([list(clustering).count(i) * len([1 for j in range(len(clustering))
                                                     if (clustering[j] != i and y_pred[
            j] != i)]) / len([1 for j in clustering
                              if j != i]) for i in np.unique(clustering)])) / len(clustering)

    return [acc, prec, rec, spec]


def classify(idx_train, idx_test, clustering_train, clustering_test, indexes_set, distances_list=None):
    y_pred_train = neigbhor_predict(distances_list[idx_train:idx_train+len(clustering_train):1, :], False)
    y_pred_test = neigbhor_predict(distances_list[idx_test:idx_test+len(clustering_test):1, :], False)
    results_train = assess(clustering_train, y_pred_train)
    results_test = assess(clustering_test, y_pred_test)
    confusion = metrics.confusion_matrix(clustering_test, y_pred_test)
    return results_train, results_test, confusion
