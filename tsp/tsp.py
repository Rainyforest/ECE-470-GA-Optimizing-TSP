"""
Modelling the TSP problems
    Input:
        A set of pts            (0,1,..., n-1)
        Distance btw pts        dist(p1, p2)

    Output:
        Shortest path
        Shortest path length
"""
import itertools
import random
import matplotlib.pyplot as plt


class Pt(complex):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TSP:
    def __init__(self, pts_size):
        self.pts = self.rand_pts(pts_size)

    # The algorithm for finding the shortest path, replaceable
    def min_path(self):
        paths = self.all_paths(self.pts)
        return min(paths, key=self.path_length)

    def path_length(self, path):
        return sum(self.dist(path[x], path[x - 1]) for x in range(len(path)))

    # Distance function btw 2 points, replaceable (e.g. sphere distance, navigation distance)
    @staticmethod
    def dist(p, q):
        return abs(p - q)

    @staticmethod
    def all_paths(pts):
        return itertools.permutations(pts)

    @staticmethod
    def rand_pts(n, width=900, height=600, seed=42):
        # Make a set of n cities, each with random coordinates within a (width x height) rectangle.
        random.seed(seed * n)
        return [Pt(random.randrange(width), random.randrange(height))
                for x in range(n)]

    @staticmethod
    def plot_path(path, style='bo-'):
        plt.figure()
        points = list(path) + [path[0]]
        plt.plot([p.x for p in points], [p.y for p in points], style)
        plt.axis('scaled')
        plt.axis('off')
        plt.show()


# Test code

tsp = TSP(9)
tsp.plot_path(tsp.min_path())
