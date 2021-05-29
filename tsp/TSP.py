"""
Modelling the TSP problems
    Input:
        A set of pts            (0,1,..., n-1)
        Distance btw pts        dist(p1, p2)

    Output:
        Shortest path
        Shortest path length
"""
import random

import matplotlib.pyplot as plt

from GeneticAlgorithm import TSPOptimizer


class TSP:
    def __init__(self, pts_size):
        self.pts = self.rand_pts(pts_size)  # [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3)]

    # The algorithm for finding the shortest path, replaceable
    def min_path(self):
        nodes = self.pts
        tsp = TSPOptimizer(population_size=100, individual_size=len(nodes), crossover_prob=0.65, mutation_prob=0.035,
                           n_iter=1000, nodes=nodes)
        return tsp.evolve()

    @staticmethod
    def rand_pts(n, width=900, height=600):  # seed =
        # Make a list of n cities, each with random coordinates within a (width x height) rectangle.
        # random.seed(seed * n)
        return [(random.randrange(width), random.randrange(height))
                for x in range(n)]

    @staticmethod
    def plot_path(path, style='bo-'):
        plt.figure()
        points = path + [path[0]]
        plt.plot([p[0] for p in points], [p[1] for p in points], style)
        plt.axis('scaled')
        plt.axis('off')
        plt.show()


# Test code
tsp = TSP(1000)
tsp.plot_path([tsp.pts[x] for x in tsp.min_path()])
