# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 01:48:30 2022

@author: suraj
"""

import random

import numpy as np
from fcm_estimator import FCMeansEstimator

class Genetic_Algo:
    def __init__(self,svr,data):
        self.pop_size  = 5
        self.population = self.__create_population()
        self.data = data
        self.svr = svr
        self.y = self.svr.estimate_missing_value()
        
        
    def __create_population(self):
        return [[random.randrange(2, 10), random.uniform(1.1, 15.0)] for _ in range(self.pop_size)]
    
    @classmethod
    def mutated_genes(self):
        
        gene = [random.randrange(2, 5), random.uniform(1.1, 15.0)]
        return gene
    
    def cal_fitness(self,param):
        
        fcm_estimator = FCMeansEstimator(param[0], param[1], self.data)
        x = fcm_estimator.estimate_missing_values()
        f = np.power((x - self.y), 2).sum()
        return f
    
    def mate(self, ch1,ch2):
        '''
        Perform mating and produce new offspring
        '''
 
        # chromosome for offspring
        child_chromosome = []
          
 
            # random probability 
        prob = random.random()
 
            # if prob is less than 0.45, insert gene
            # from parent 1
        if prob < 0.45:
            child_chromosome.append(ch1)
 
            # if prob is between 0.45 and 0.90, insert
            # gene from parent 2
        elif prob < 0.90:
            child_chromosome.append(ch2)
 
            # otherwise insert random gene(mutate),
            # for maintaining diversity
        else:
            child_chromosome.append(self.mutated_genes())
 
        # create new Individual(offspring) using
        # generated chromosome for offspring
        return child_chromosome[0]
    
    def run(self):
        for _ in range(5):
            # sort the population in increasing order of fitness score
            self.population = sorted(self.population, key = self.cal_fitness)
  
            # Otherwise generate new offsprings for new generation
            new_generation = []
 
            # Perform Elitism, that mean 10% of fittest population
            # goes to the next generation
            s = int((10*self.pop_size)/100)
            new_generation.extend(self.population[:s])
 
            # From 50% of fittest population, Individuals
            # will mate to produce offspring
            s = int((90*self.pop_size)/100)
            for _ in range(s):
                parent1 = random.choice(self.population[:40])
                parent2 = random.choice(self.population[:40])
                child = self.mate(parent1, parent2)
                new_generation.append(child)
 
            self.population = new_generation
            
        self.population = sorted(self.population, key = self.cal_fitness)
        return self.population[0][0], self.population[0][1]
 
        
