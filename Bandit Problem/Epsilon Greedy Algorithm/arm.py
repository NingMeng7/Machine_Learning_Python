import random
import numpy as np


class Arm(object):
    """the arm of bandit problem with normal distribution"""
    
    def __init__(self, mean, variance, initial_value):
        self.mean = mean
        self.variance = variance
        self.time_of_selected = 0
        self.estimate_value = initial_value
    
    def change_distribution(self):
        """Change the distribution of a arm by adding a random increment to the mean of the arm"""
        self.mean += np.random.normal(0, 0.1)
       
    
    def generate_reward(self):
        """Return a reward based on the current distribution of the arm"""
        return np.random.normal(self.mean, self.variance)