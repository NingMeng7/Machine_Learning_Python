import math
import random
from Individual import Individual

class Genetic_Algorithm(object):
    def __init__(self, length, quantity, kill_rate, mutation_rate):
        """
        length: the length of chromosome(binary encoding)
        population: the number of parrent+children in each generation
        """
        self.length = length
        self.quantity = quantity
        self.population = self.initial()

        self.current_quantity = quantity
        self.mutation_rate = mutation_rate
        self.kill_rate = kill_rate


    def initial(self):
        return [Individual(self.length, 0.01) for i in range(self.quantity)]
        

    def evolve(self):
        # Stimulate the evolution of the population
        paraents = self.kill()
        self.crossover()
        self.mutation()
        

    def kill(self):
        # Kill part of the current individuals
        fitness = [self.population[k].value for k in range(self.quantity)]
        sorted_fitness = sorted(fitness)
        for k in range(int(self.quantity * self.kill_rate)):
            index = fitness.index(sorted_fitness[0])
            del self.population[index]
            del sorted_fitness[0]
            del fitness[index]

        self.current_quantity = self.quantity - int(self.quantity * self.kill_rate)
    
        
    def crossover(self):
        number = self.quantity - self.current_quantity # The number of needed children
        
        while number >= 1:
            male = random.randint(0, self.current_quantity - 1)
            female = random.randint(0, self.current_quantity - 1)
            if male == female:
                continue

            cross_point = random.randint(0, self.length)
            child = self.population[male].chromosome[0:cross_point] + self.population[female].chromosome[cross_point:self.length]
            self.population.append(Individual(self.length, self.mutation_rate, child))
            number -= 1

        self.current_quantity = self.quantity


    def mutation(self):
        for k in range(self.quantity):
            self.population[k].genic_mutation()