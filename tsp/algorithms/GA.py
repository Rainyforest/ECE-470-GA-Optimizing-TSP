"""
Encoding
    - Binary Encoding
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
Mutation
    - Randomly select gene and change their values
Terminate Conditions (one of them)
    - No improvement to the population fitness
    - After a predefined number of generations
    - When we reach a specific fitness value
"""

import random

# This should encode a path to a string
def encoding():
    pass

def population_initialization():
    pass

def reproduce(x, y):
    pass

def selection(population):
    pass

def crossover():
    pass

# A function to provide small chance of randomness for mutation
def is_mutate():
    pass

def mutation(individual):
    if is_mutate(): # small chance
        # do something
        pass
    else: 
        # do not mutate
        pass

def fitness_fn(individual):
    pass

# Function to decide if the generation reproduces satisfying individual
# and could terminate the training
def is_optimal(population):
    # iterate through the new population and call fitness function 
    pass


  
def GA(population, fitness_fn):
    while not is_optimal():
        new_population = {}
        # generate a group of new generation
        for i in range(len(population)):
            # parents selection
            x = selection(population)
            y = selection(population)
            # crossover
            child = crossover(x,y)
            # mutation
            child = mutation(child)
            new_population.add(child)
        population = new_population
    return population
        
