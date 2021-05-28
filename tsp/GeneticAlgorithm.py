import numpy as np

"""
Encoding
    - Binary Encoding
      Reference: https://www.researchgate.net/publication/311709380_A_New_Approach_using_Binary_Encoding_to_Solve_Travelling_Salesman_Problem_with_Genetic_Algorithms
    - Value Encoding
    - Permutation Encoding 
    - Tree Encoding
Population initialization
    - Random Initialization
    - Heuristic Initialization
Population Models
    - Incremental
    - Generational
Fitness Function
    - Requirements: Efficient & Sensitive
Selection
    - Proportionate Selection
        - Roulette Wheel Selection
         （Proportional to the Fitness）
        - Random Selection
        - Tournament Selection
        - Rank Selection
Crossover
    - Two selected individuals generate one or more offspring
        - Partially mapped crossover (PMX)
          Reference: https://user.ceng.metu.edu.tr/~ucoluk/research/publications/tsp
Mutation
    - Randomly select gene and change their values
        - Dynamic Mutation
          Reference: https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050918X00076/1-s2.0-S1877050918306100/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDcaCXVzLWVhc3QtMSJGMEQCIHcMhzfvpB6iMZmPyjTBh5FbXbw88vjMxp4rRy6rRGWSAiB%2Ffk7oIYpFbgVt%2BP2nI1Wd%2BP0c%2Bg1HqJRpWrOsttb5riq9AwjQ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAMaDDA1OTAwMzU0Njg2NSIMqNJID0NqrsBoU85LKpEDqjiZ4UZU5yODqdHNTnfSXueC%2BnrbH9E0tQSjdW9HLUhukWwCRVPL3UBEIHQdwBTY713rycws9UhwdqtoEiQkfJ0f7WZm%2Fnq1bUrSi69hX4J1qxmHibvltXjpGptqGRQ6ZNe2o%2Bqm3zf4MmKmOJhE%2Bi6UyrgpL3nV5JtQz7%2BxLG9ZJHKKSriSVf%2FO4OiWID375CDiKX1leEXoaAEZMTQwhP93qQOjQZs7aMCaOJIDD5y06bz3CPRHlhFUY5G3Jw95gq1sL8uFgVVWUA32vvrpL5T2mweQ7vAK3IfDdsh3u6mYG2G0lnncfYSyYZRwnx6aFMomdb5%2BL9wHCVbpuz5f3vfR%2BRORW7JlolQ%2FvmT%2Bcj%2FnhG7Gy%2BOxoW3OPM299pN3RyzWdh0y%2B%2BIgA0ttdtWzEj7IHEHJIKgMZdBu%2FzePk3H6%2Fy6dJY0tAhwog2c3HiHVsqBF%2F3uT2L7%2BqyevcVTT4i1Fi3mLdMx64HX4h%2FRg%2Bvxcz2WT8KYdLbgATUQzKNf9h4G8OpQ%2F7yJy1SKe8X8%2BdAkwzfqnhQY67AF%2BKCIty6yiQPXfOe9p06swrrSsJAUYaj5VfZbq3TKxegPDWDHlDcwEDclBvCei0YBsw8Z%2BGOABvINKtJc1R18iH3tuYkf41YBSi8Y%2BY5pWZeaE9t2rEEQd1XFh9%2Bao3pnLw2rqtINeovGojfny9Dm6Qxr5W1PKQNm214C9nj8ubqvBjRAFNukDua0XEiR69%2BYBiJISoI7LSXXVV%2BKv2kvBw5YlYBWCS1bt4pIDWsljPvvZTS5r8xv3XTUJKwEjq6J%2FsOyP8FJhsFluG9f3OpaJzwbCJ30EpFLQ7QicTMhW6vTlpaQohTHF9DHZdQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210523T080607Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY64V3GOMB%2F20210523%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=89b0ed923d61a583e04d9b13a46160e442d26a4068a6b407cccba39e1a7b836b&hash=dd7a366f15666f23ce1629f7a62b5428ce74ece993aafaf2f3fabc64a8f277ef&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050918306100&tid=spdf-e787cdf3-43a1-4d04-bc61-470b002cbc4c&sid=021b3ffa5516184f990b316-ef18f218934egxrqa&type=client
Terminate Conditions (one of them)
    - No improvement to the population fitness
    - After a predefined number of generations
    - When we reach a specific fitness value
"""

class GeneticAlgorithm:

    def __init__(self, population_size, individual_size, crossover_prob, mutation_prob, n_iter, maximize, fitness):
        self.population_size = population_size
        self.individual_size = individual_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_iter = n_iter
        self.maximize = maximize
        self.fitness = fitness if maximize else lambda x: -fitness(x)

    def encode(self):
        return None

    def decode(self, population):
        return population

    def select(self, population, fitness):
        return population

    def crossover(self, population1, population2):
        return population1

    def mutate(self, population):
        return population

    def positive_fitness(self, population):
        population = self.decode(population)
        fitness_score = self.fitness(population)
        return fitness_score - fitness_score.min()

    def reproduce(self, population):
        n_crossover = int(np.round(self.crossover_prob * self.population_size))
        offspring = population.copy()
        np.random.shuffle(offspring)
        offspring[:n_crossover] = self.crossover(population[:n_crossover], offspring[:n_crossover])
        return offspring

    def evolve(self):
        population = self.encode()
        optimal = None
        for _ in range(self.n_iter):
            fitness = self.positive_fitness(population)
            population = self.select(population, fitness)
            population = self.reproduce(population)
            population = self.mutate(population)
            optimal = population[np.argmax(fitness)]
        return optimal


class UnaryFunctionOptimizer(GeneticAlgorithm):

    def __init__(self, population_size, individual_size, crossover_prob, mutation_prob, n_iter, maximize, obj_func, interval):
        super().__init__(population_size, individual_size, crossover_prob, mutation_prob, n_iter, maximize, obj_func)
        self.interval = interval

    def encode(self):
        population = np.random.randint(low=0, high=2, size=(self.population_size, self.individual_size)).astype(np.int8)
        return population

    def decode(self, population):
        population = (population.dot(np.power(2, np.arange(self.individual_size)[::-1])) / np.power(2, self.individual_size) - 0.5) * \
                     (self.interval[1] - self.interval[0]) + 0.5 * (self.interval[0] + self.interval[1])
        return population

    def select(self, population, fitness):
        fitness = fitness + 1e-8
        chosen = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True, p=fitness / fitness.sum())
        return population[chosen]

    def crossover(self, population1, population2):
        cxpoints = np.random.randint(0, 2, self.individual_size).astype(np.bool8)
        population1[:, cxpoints] = population2[:, cxpoints]
        return population1

    def mutate(self, population):
        mtpoints = np.random.choice([0, 1], size=self.population_size * self.individual_size, p=[1 - self.mutation_prob, self.mutation_prob]).reshape(self.population_size, self.individual_size)
        population = (population + mtpoints) % 2
        return population


class TSPOptimizer(GeneticAlgorithm):

    def __init__(self, population_size, individual_size, crossover_prob, mutation_prob, n_iter, nodes):
        super().__init__(population_size, individual_size, crossover_prob, mutation_prob, n_iter, False, self.fitness)
        self.nodes = nodes

    def fitness(self, population):
        x_coord = np.array([list(map(lambda x: self.nodes[x][0], perm)) for perm in population])
        y_coord = np.array([list(map(lambda x: self.nodes[x][1], perm)) for perm in population])
        return np.sum(np.sqrt(np.square(np.diff(x_coord)) + np.square(np.diff(y_coord))), axis=1)

    def encode(self):
        population = np.array([np.random.permutation(len(self.nodes)) for _ in range(self.population_size)])
        return population

    def decode(self, population):
        population = np.array([perm + [perm[0]] for perm in population.tolist()])
        return population

    def select(self, population, fitness):
        fitness = fitness + 1e-8
        idx = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True, p=fitness / fitness.sum())
        return population[idx]

    def crossover(self, population1, population2):
        for i in range(population1.shape[0]):
            population1[i], _ = self.pmx(population1[i], population2[i])
        return population1

    def pmx(self, individual1, individual2):
        p1, p2 = [0] * self.individual_size, [0] * self.individual_size
        for i in range(self.individual_size):
            p1[individual1[i]] = i
            p2[individual2[i]] = i
        cxpoint1 = np.random.randint(0, self.individual_size)
        cxpoint2 = np.random.randint(0, self.individual_size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        for i in range(cxpoint1, cxpoint2):
            temp1 = individual1[i]
            temp2 = individual2[i]
            individual1[i], individual1[p1[temp2]] = temp2, temp1
            individual2[i], individual2[p2[temp1]] = temp1, temp2
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
        return individual1, individual2

    def mutate(self, population):
        population_copy = population.copy()
        n_mutation = int(np.round(self.mutation_prob * self.population_size))
        mtpoint1, mtpoint2 = np.random.choice(self.individual_size, 2, replace=False)
        population[:n_mutation, mtpoint1] = population_copy[:n_mutation, mtpoint2]
        population[:n_mutation, mtpoint2] = population_copy[:n_mutation, mtpoint1]
        return population

# Test code
if __name__ == '__main__':
    obj_func = lambda x: np.sin(x)
    interval = [-2, 2]
    optimizer = UnaryFunctionOptimizer(population_size=256, individual_size=16, crossover_prob=0.9, mutation_prob=0.01, obj_func=obj_func, n_iter=1000, maximize=False, interval=interval)
    opt_individual = optimizer.evolve()
    opt_decoding = optimizer.decode(opt_individual)
    opt_obj = obj_func(opt_decoding)
    print(opt_individual, opt_decoding, opt_obj)

    nodes = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3)]
    tsp = TSPOptimizer(population_size=256, individual_size=len(nodes), crossover_prob=0.9, mutation_prob=0.01, n_iter=1000, nodes=nodes)
    opt_individual = tsp.evolve()
    opt_decoding = np.array(opt_individual.tolist() + [opt_individual[0]])
    x_coord = [nodes[node][0] for node in opt_decoding]
    y_coord = [nodes[node][1] for node in opt_decoding]
    opt_obj = np.sum(np.sqrt(np.square(np.diff(x_coord)) + np.square(np.diff(y_coord))))
    print(opt_individual, opt_decoding, opt_obj)
