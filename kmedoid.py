import pickle
import numpy as np
import random
import argparse
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


class KMedoid:
        def __init__(self, data, num_cluster=2, max_iteration=10, max_no_change_iteration = 5,
                        classes = None, verbose=True):

                self.data = data
                self.num_cluster = num_cluster
                self.max_iteration = max_iteration
                self.max_no_change_iteration = max_no_change_iteration
                self.cluster_candidates = np.arange(0, max(np.shape(data)))
                self.classes = classes
                self.verbose = verbose
                
        def clusterize(self, dist_mat, medoids):
                candidates = self.prepare_candidates(self.cluster_candidates, medoids)
                row_dists = []
                for m in medoids:
                        row_dists.append(dist_mat[m, candidates])
                clusters = self.create_cluster_dict(medoids)

                for c in candidates:
                        row = dist_mat[c, medoids]
                        cluster_index = row.tolist().index(min(row))
                        clusters[medoids[cluster_index]]['cluster'].append(c)
                
                for key in clusters.keys():
                        clusters[key]['cost'] = sum(dist_mat[key][clusters[key]['cluster']])

                return clusters

        def make_clusters(self):
                mat = self.normalize_confusion_matrix(self.data)
                dist = self.distance_matrix(np.array(mat))
                medoids = self.select_medoids(self.num_cluster, dist.shape)
                clusters = self.create_cluster_dict(medoids)
                clusters = self.clusterize(dist, medoids)
                config_cost = self.get_config_cost(clusters)
                temp_medoids = medoids.copy()
                config_change_counter = 0
                if self.verbose != 0:
                        print("-----------------------num_clusters = {}--------------------------".format(self.num_cluster))                
                for k in range(self.max_iteration):
                        temp_medoids[0] = self.select_medoids(1, dist.shape, medoids)[0]
                        if self.verbose != 0:
                                print('checking for medoid: ', temp_medoids)
                        temp_clusters = self.clusterize(dist, temp_medoids)
                        temp_config_cost = self.get_config_cost(temp_clusters)
                        if (temp_config_cost < config_cost):
                                medoids = temp_medoids.copy()
                                clusters = temp_clusters
                                config_cost = temp_config_cost
                                config_change_counter = 0
                        else:
                                config_change_counter = config_change_counter + 1
                        if self.verbose == 1:
                                print('Iteration {}: {} ({:.2f}) {}'.format(k, [clusters[i]['cluster'] for i in clusters.keys()], config_cost, medoids))
                        elif self.verbose == 2 and self.classes is not None:
                                cluster_values = [clusters[i]['cluster'] for i in clusters.keys()]
                                print('iteration {}: {} ({:.2f}) {}\n'.format(k, [self.classes[c].tolist() for c in cluster_values], config_cost, medoids))
                        if (config_change_counter == self.max_no_change_iteration):
                                break
                return clusters
                
        def get_config_cost(self, clusters):
                return sum([clusters[i]['cost'] for i in clusters.keys()])

        def prepare_candidates(self, candidates, medoids):
                for m in medoids:
                        candidates = candidates[candidates != m]
                return candidates

        def select_medoids(self, num, data_shape, medoids=None):
                '''
                returns random medoids
                '''
                population = np.arange(max(data_shape))
                if medoids:
                        population = np.delete(population, medoids)
                return random.sample(population.tolist(), num)

        def normalize_confusion_matrix(self, matrix):
                return [[round(x/sum(r), 5) for x in r] for r in matrix]
        
        def distance_matrix(self, n_matrix):
                '''
                n_matrix must be a numpy array
                calculates distance matrix.
                '''
                # return np.reshape([1 - n_matrix[x][y] for (x, y) in np.ndindex(np.shape(n_matrix))], np.shape(n_matrix))
                return 1 - n_matrix
        
        def create_cluster_dict(self, medoids):
                return {i: {'cluster': [i], 'cost': 0} for i in medoids}


def display_cluster(x):
        for i in range(len(x)):
                for key in x[i].keys():
                        print(x[i][key]['cluster'], " : cost" , x[i][key]['cost'])
                print("--------")

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_clusters', type=int, default=3)
        parser.add_argument('--max_iteration', type=int, default=100)
        parser.add_argument('--max_no_change_iteration', type=int, default=10)
        parser.add_argument('--verbose', type=int, default=1)
        args = parser.parse_args()
        print(type(args.num_clusters), args.max_iteration)



        data = pickle.load(open('confusion_matrix', 'rb'))
        confusion_mat = data['mat']
        # classes=np.arange(10)
        # confusion_mat = [[828,	13,	12,	11,	18,	0,	2,	4,	85,	27],
        #                 [10,	910,	0,	5,	1,	1,	0,	1,	11,	61],
        #                 [47,	1,	708,	64,	88,	14,	63,	4,	8,	3],
        #                 [3,	4,	16,	768,	33,	93,	50,	19,	4,	10],
        #                 [10,	0,	39,	43,	788,	12,	57,	43,	6,	2],
        #                 [2,	0,	10,	137,	29,	777,	8,	33,	0,	4],
        #                 [7,	2,	10,	54,	29,	7,	888,	1,	1,	1],
        #                 [24,	2,	14,	39,	76,	17,	4,	818,	2,	4],
        #                 [27,	13,	0,	7,	3,	0,	3,	0,	933,	14],
        #                 [19,	64,	1,	7,	2,	1,	1,	0,	18,	887]]
        classes = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", " frog", "horse", "ship", "truck"])
        x = list()
        for i in range(1,10):
                #param1: confusion matrix to clusterize, param2: number of clusters to make
                km = KMedoid(confusion_mat, i, max_iteration=args.max_iteration, max_no_change_iteration=args.max_no_change_iteration,
                        classes=classes, verbose=2) 
                clusters = km.make_clusters() # returns clusters
                x.append(clusters)
                # print(classes[clusters['clusters']])
        display_cluster(x)
        # print(clusters)
        print("END")
        
        # linked = linkage([[0,1,2,3,4,5,6,7,8,9],
        # [1,2,3,4,5,5,6,7,8,9],
        # [1,2,3,4,4,4,5,5,6,6],
        # [12,32,3,4,23,4,3,34,3,5]
        # ] , 'single')

        # labelList = range(0, 10)

        # plt.figure(figsize=(10, 7))
        # dendrogram(linked,
        #         orientation='top',
        #         labels=labelList,
        #         distance_sort='descending',
        #         show_leaf_counts=True)
        # plt.show()

#[1,8,9]
#[7,0,2, 3, 4, 5, 6]