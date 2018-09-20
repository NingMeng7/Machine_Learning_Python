import random
import math
class Individual(object):
    """individual of the population"""
    def __init__(self, length, mutation, chromosome = ""):
        self.length = length
        if chromosome == "":
            self.chromosome = self.generate_chromosome(self.length)
        else:
            self.chromosome = chromosome
        self.caculate_value()
        self.mutation_rate = mutation

    def generate_chromosome(self, length):
        # Generate a random chromosome
        string = ""
        seed = "01"
        
        for i in range(length):
            string += str(random.choice(seed))
        
        return str(string)


    def caculate_value(self):
        # Caculate the fitness value of the individual
        x = self.decode()
        self.value = x + 10*math.sin(5*x) + 7*math.cos(4*x)

    
    def decode(self):
       # map the binary choromosome to a real number
       return int(self.chromosome, 2) * (9 - 0) / (2**self.length - 1)


    def genic_mutation(self):
        if random.random() < self.mutation_rate:
            index = random.randint(0, self.length-1)
            temp = list(self.chromosome)
            
            if self.chromosome[index] == "0":
                temp[index] = "1"
                self.chromosome = "".join(temp)
            else:
                temp[index] = "0"
                self.chromosome = "".join(temp)
        self.caculate_value()