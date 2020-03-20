import os
import sys
import time
import random
import decimal

#Compute the total path length of a specified tour
def path_cost(sample_tour):
    cost = 0 
    for city in range(0,len(sample_tour)-1):
        a = sample_tour[city]
        b = sample_tour[city+1]
        cost += distance_matrix[a][b]
    return cost 

#Heuristic cost function; the path cost of the current tour is used
def heuristic(sample_tour):
    if path_cost(sample_tour) == 0:
        return 0
    return -1 * path_cost(sample_tour)

#Cooling schedule; update value of T
#In the basic implementation of the algorithm, a linear increment is used
def cooling_schedule(t,increment): 
    #Linear
    T = t - increment  
    return T

#In the basic implementation of the algorithm, a single successor operator is used
def find_successor(sample_tour):
    #Reverse the ordering of a random subset of the tour
    next_successor = sample_tour[0:-1].copy()
    i = random.randrange(len(next_successor)-1)
    j = random.randrange(len(next_successor)-1)
    while True:
        if i == j:
            find_successor(sample_tour)
        if i < j:    
            subset = next_successor[i:j+1]
            z=i
        else: 
            subset = next_successor[j:i+1]
            z=j
        subset.reverse()
        for entry in subset:
            next_successor[z] = entry
            z+=1
        next_successor.append(next_successor[0])
        return next_successor

#Probability function; depends on values of delta E and T
def probability_calc(deltaE,T):
    calc = abs(deltaE)/T
    probability = decimal.Decimal(calc).exp()
    probability = 1/probability
    return probability

def simulated_annealing(t, cooling_increment_factor):
    #Generate a random initial tour
    initial_tour = [i for i in range(num_cities)]
    random.shuffle(initial_tour)
    #Ensure that the tour ends by returning to the start city
    initial_tour.append(initial_tour[0])
    current = initial_tour
    T = t
    while True:
        T = cooling_schedule(T,cooling_increment_factor)
        if T <= 1:
            break
        else:
            #Randomly choose a successor of the current state
            successor = find_successor(current)
            #Calculate the difference delta E according to a specified heuristic cost function
            deltaE = heuristic(successor) - heuristic(current)
            #If delta E is positive, the selected successor state becomes the new current state
            if deltaE >= 0:
                current = successor
            #If delta E is negative, the successor replaces the current state only with specified probability
            else: 
                #Calculate the value of the probability function
                probability = probability_calc(deltaE, T)
                #Randomly generate a value between 0 and 1
                select_random = random.random()
                #Check if probability condition holds
                if probability >= select_random:
                    current = successor
    tour = current
    return tour

#Generate tour for supplied input data
full_tour = simulated_annealing(85,1.00010179)
tour_length = path_cost(full_tour)
tour = full_tour[:-1]
