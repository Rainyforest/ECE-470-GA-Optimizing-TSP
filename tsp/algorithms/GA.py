"""
Encoding
    - Binary Encoding
      Reference: https://www.researchgate.net/publication/311709380_A_New_Approach_using_Binary_Encoding_to_Solve_Travelling_Salesman_Problem_with_Genetic_Algorithms
    - Value Encoding
    - Permutation Encoding (choose)
    - Tree Encoding
Population initialization
    - Random Initialization
    - Heuristic Initialization
Population Models
    - Incremental
    - Generational (currently choose)
Fitness Function
    - Requirements: Efficient & Sensitive
Selection
    - Proportionate Selecton
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

import random

class GA:
        
    # This should encode a path to a string
    def encoding(self, path):
        pass

    def decoding(self, string):
        pass

    def population_initialization():
        pass

    def selection(self, population):
        pass

    def crossover(self, x, y):
        pass

    # A function to provide small chance of randomness for mutation
    def is_mutate(self):
        pass

    def mutation(self, individual):
        if self.is_mutate(): # small chance
            # do something
            pass
        else: 
            # do not mutate
            pass

    def fitness_fn(self, individual):
        pass

    # Function to decide if the generation reproduces satisfying individual
    # and could terminate the training
    def is_optimal(self, population):
        # iterate through the new population and call fitness function 
        pass


    
    def evolve(self, population):
        while not self.is_optimal():
            new_population = {}
            # generate a group of new generation
            for i in range(len(population)):
                # parents selection
                x = self.selection(population)
                y = self.selection(population)
                # crossover
                child = self.crossover(x,y)
                # mutation
                child = self.mutation(child)
                new_population.add(child)
            population = new_population
        return population
        
