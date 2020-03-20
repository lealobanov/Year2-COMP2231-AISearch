import os
import sys
import time
import random

#Generate initial population by randomizing order of cities in sample tour 
def population(pop_size):
    #Generate a sample city tour
    sample = []
    for i in range(0,num_cities):
        sample.append(i)
    sample.append(sample[0])
    population = []
    #Shuffle sample tour and add to initial population
    for i in range(0,pop_size):
        randomized_tour = sample[0:-1].copy()
        random.shuffle(randomized_tour)
        randomized_tour.append(randomized_tour[0])
        population.append(randomized_tour[:])
    return population

#Calculate the total cost of a given tour
def path_cost(sample_tour):
    cost = 0 
    #sample_tour.append(sample_tour[0])
    for city in range(0,len(sample_tour)-1):
        a = sample_tour[city]
        b = sample_tour[city+1]
        cost += distance_matrix[a][b]
    return cost 

#Define fitness function (aim to maximize); treat fitness as inverse of tour cost
def fitness(population):
    fitnesses = []
    for tour in population:
        tour_cost = path_cost(tour)
        if tour_cost != 0:
            tour_fitness = 100*(1/tour_cost)
        else: 
            tour_fitness = 101
        fitnesses.append(tour_fitness)
    return fitnesses

#Find and store the fittest individual in a given population
def fittest_individual(population):
    fitnesses = fitness(population)
    sorted_fitnesses = sorted(fitnesses,reverse=True)
    fittest_member = sorted_fitnesses[0]
    best_tour_index = fitnesses.index(fittest_member)
    best_tour = population[best_tour_index]
    return best_tour

#Select 2 parents from the population to breed; fitter individuals are more likely to be selected 
#In the basic implementation: fitness proportionate selection by roulette wheel method
def selection(population):
    probabilities = []
    tour_prob = []
    sum_fitnesses = sum(fitness(population))
    for entry in population:
        tour_fitness = 100*(1/path_cost(entry))
        probability = tour_fitness/sum_fitnesses
        probabilities.append(probability)
        tour_prob.append(probability)
    sorted_probabilities = sorted(probabilities)
    #Randomly generate a value between 0 and 1
    selected = random.random()
    current = 0
    for p in probabilities:
        current += p
        if current >= selected:
            selected_p = p
            break
    tour = tour_prob.index(selected_p)
    selected_tour = population[tour]
    return selected_tour

#Crossover method: ordered crossover
def crossover(a,b):
    child_p1 = []
    parent1 = a[:-1]
    parent2 = b[:-1]
    #Randomly select 2 positions at which crossover will occur
    j = random.randrange(len(parent1))
    k = random.randrange(len(parent2))
    start_index = min(j,k)
    end_index = max(j,k)
    #Initialize child
    for i in range(num_cities):
        child_p1.append('-')
    #Transfer segment at indices i through j in parent 1
    for i in range(start_index, end_index+1):
        child_p1[i] = parent1[i]
    position1 = 0 
    position2 = 0 
    #Fill remaining positions with cities in the order they appear in parent 2
    while position1 < len(parent1):
        if child_p1[position1] != '-':
            position1 += 1
        else: 
            if parent2[position2] not in child_p1:
                child_p1[position1] = parent2[position2]
                position1 +=1
                position2 +=1
            elif parent2[position2] in child_p1: 
                position2 +=1
    child_p1.append(child_p1[0])
    return child_p1

#Mutation method: swap cities at 2 randomly specified indices
#Probability of mutation occurring is supplied as an algorithmic parameter
def mutation(child, probability):
    mut_child = child[0:-1].copy()
    i = random.randrange(len(mut_child))
    j = random.randrange(len(mut_child))
    if i == j:
        mutation(child, probability)
    select_random = random.random()
    if probability >= select_random:
        mut_child[i], mut_child[j] = mut_child[j], mut_child[i]
    mut_child.append(mut_child[0]) 
    return mut_child

#Produce next generation
def next_generation(current_generation, mutation_rate):
    next_gen = []
    i=0
    while i < len(current_generation):
        child = mutation(crossover(selection(current_generation), selection(current_generation)), mutation_rate)
        next_gen.append(child[:])
        i +=1
    return next_gen

def genetic_algorithm(pop_size, generations, mutation_rate):
    #Create an initial population
    current_population = population(pop_size)
    #Identify an initial best tour; used as a basis of comparison for all future tours generated
    best_tour = fittest_individual(current_population)
    best_tour_cost = path_cost(best_tour)
    #Produce subsequent generations 
    for i in range(0, generations):
        current_population = next_generation(current_population, mutation_rate)
        #Determine the fittest individual of the new generation
        new_tour_best = fittest_individual(current_population)
        new_cost = path_cost(new_tour_best)
        #Check if the new fittest tour is better than the global best tour; if yes, update the best tour 
        if new_cost < best_tour_cost:
            best_tour =  new_tour_best
            best_tour_cost = new_cost
    return best_tour

#Call genetic algorithm with specified parameters
#Return best tour found and its associated cost
full_tour = genetic_algorithm(15,1500,0.07)
tour_length = path_cost(full_tour)
tour = full_tour[:-1]

    











    


