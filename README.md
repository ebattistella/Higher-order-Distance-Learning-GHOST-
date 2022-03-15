# GHOST

## Description
GHOST is a distance learning technique defined in [Battistella 2022](https://hal.archives-ouvertes.fr/hal-03563705/).
It relies on conditional random fields and graph properties to estimate the best distance metric for a given classification task.
This distance can then be leveraged through an adapted K-Nearest Neighbors (K-NN) approach to predict a
sample class. This algorithm also allows to determine the features importance and can be used for 
higher-order feature selection.

The basic second-order implementation provides a usual distance defined from the provided features.
This method has been published in [Komodakis 2011](https://doi.org/10.1109/ICCV.2011.6126227).

The higher-order implementation provides an extended notion of distance as defined in the article.
It offers a usual distance defined from the provided features and either a provided natural 
graph structure for the dataset or a graph generated using K-NN. 
The second order graph property leveraged is the shortest path.
The higher-order graph properties leveraged are the shortest path, the clique order, the eccentricity and
the connectivity.

## Use
The structure of both the second-order and the higher-order implementations are similar, they both include 
the files 'classify.py' and 'distance_learning.py'. The higher-order method also uses a 'graph.py' file to define the
graph notion to be used and the diverse functions to compute the needed graph properties and graph operations.
We describe next the files and functions for the higher-order case, the second-order case being built on the 
same template less the higher-order specific parameters.

- 'distance_learning.py': it provides in the main function an example on how to use the learn function to obtain the
higher-order metric and how to use it for classification. Only the 'learn' function is meant to be called, its 
parameters are:
  - clustering_list: ground-truth, list of labels for the samples for the K training sets
  - feature_list: tensor containing for each pair of samples their distance according to each metric we are learning on
  - T: number of projected subgradient rounds
  - C: coefficient for the convergence criterion, cf convergence function
  - max_it: maximum number of iterations in case convergence is not reached
  - alpha, beta, tau: coefficients of the constraints and of the regularization, to be tuned
  - speed: coefficient influencing the speed of convergence, weight the updates
  - update_name: name of the projection to be update at each iteration, in the higher-order case, the weights need to be positive
  - nn_numbers: size of the neighborhood to consider when buildind graphs based on the neighborhood
  - order: order to consider for the higher- order
  - cpu_nb_slaves: number of cpus for available for parallelization
  - regu: name of the regularization method, implemented methods are "lasso", "ridge" and "elasticnet"
  - init: possible pre-initialization of the weights for the second- order metrics (e.g. by first applying the second- order framework), empty list for no initialization
  - graphs: list of the graphs to consider for each of the K training sets, if empty the K- nearest neighbor method is used
  - cluster_metric : 0: no higher order, 1: higher-order of order "order" only, -1: cluster metric only, 2: higher-order and cluster metric
- 'classify.py': it performs the actual prediction using the higher-order distance previously learned.
The prediction function leverages a K-NN frameworks and can be used with various linkages (see the article
for more details). first the function 'to_centers_distance_matrix' has to be called with
  - distance: an int characterizing the type of distance we want to compute from the list "pearson", "spearman", 
"kendall", "euclidean", "cosine", "pearson_abs", "spearman_abs", "kendall_abs"
  - data_train: the training dataset
  - data_test: the testing dataset
  - w: the learned weights
  - nn_numbers: the number of nearest neighbors to consider
  - centers: the center points of the classes
  - clusters: the classes
  - order: the order
  - has_vectors: boolean to characterize if supplemental weighted distances between the feature vectors have to be computed
  - path: path to save the distance to
  - init: initial vector of weights
  - graph_train: graph of hte data defined on the training dataset
  - cluster_metric: distance to be used in the K-NN

  Then, the function classify can be called to perform the actual prediction and assess the results with the parameters:
  - clustering_train: labels of the training set
  - clustering_text: labels of the testing set
  - indexes_set: indexes of the distance matrix on which the classification has ot be performed
  - distances_list: previously learned distances
- 'graph.py': it defines the graph class based on networkx functions. 
