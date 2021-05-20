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
    x = property(lambda (self): self.real)
    y = property(lambda (self): self.imag)

    def __str__(self):
        return '(%g, %g)' % (self.real, self.imag)

    def __repr__(self):
        return str(self)


class TSP:
    # Distance function btw 2 points, replaceable (e.g. sphere distance, navigation distance)
    def dist(self, p, q):
        return abs(p - q)

    @staticmethod
    def all_paths(pts):
        return itertools.permutations(pts)

    # The algorithm for finding the shortest path, replaceable
    def min_path(self, pts):
        paths = self.all_paths(pts)
        return min(paths, key=self.path_length)

    def path_length(self, path):
        return sum(self.dist(path[x], path[x - 1]) for x in range(len(path)))


def rand_pts(n, width=900, height=600, seed=42):
    # Make a set of n cities, each with random coordinates within a (width x height) rectangle.
    random.seed(seed * n)
    return frozenset(Pt(random.randrange(width), random.randrange(height))
                     for x in range(n))


def plot_path(path, style='bo-'):
    plt.figure()
    points = list(path) + [path[0]]
    plt.plot([p.x for p in points], [p.y for p in points], style)
    plt.axis('scaled')
    plt.axis('off')
    plt.show()


# Test code
random_set = rand_pts(7)
tsp = TSP()
plot_path(tsp.min_path(random_set))
