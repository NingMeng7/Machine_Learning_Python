from arm import Arm
import random
import numpy as np


class Sample_Average(object):
    """The weighted average algorithm with a step-size parameter"""

    def __init__(self, arms, epsilon):
        self.epsilon = epsilon
        self.arms = arms
        # 初始值的设定有时候会带来bias，然而，这种bias是有用的，例如，以下的过度乐观的先验估计将会鼓励exploration
        # for k in range(len(arms)):
           # self.values[k] = 100   
           
   
    #  Select an arm by greedy algorithm
    def Action(self):
        if random.random() > self.epsilon:
            arm_index = self.get_best_arm()
        else:
            arm_index = self.get_random_arm()
        
        self.arms[arm_index].time_of_selected += 1
        
        reward = self.get_reward(arm_index)
        self.update(arm_index, reward)
    

    #  Update the value function of the selected arm
    def update(self, arm_index, reward):
        self.arms[arm_index].estimate_value += 1 / self.arms[arm_index].time_of_selected * (reward - self.arms[arm_index].estimate_value)


    #  Get a random action
    def get_random_arm(self):
        return random.randrange(len(self.arms))


    #  Get the best arm from current knowledge
    def get_best_arm(self):
        best_arm = self.arms[0].estimate_value
        best_arm_index = 0
        
        for k in range(len(self.arms)):
            if self.arms[k].estimate_value > best_arm:
                best_arm = self.arms[k].estimate_value
                best_arm_index = k
        
        return best_arm_index


    #  Get reward given a action
    def get_reward(self, arm_index):
        return self.arms[arm_index].generate_reward()


    #  Illustrate the result of algorithm
    def print_result(self):
        print("#order: ", "  Mean \t     Times of selection      \t Estimate value") 

        for k in range(len(self.arms)):
            print("#", k, ": ", self.arms[k].mean, '\t', self.arms[k].time_of_selected, '\t \t ', self.arms[k].estimate_value) 
