import os
import sys
import time
import random
import decimal
import math

#Compute the total path length of a specified tour
def path_cost(sample_tour):
    cost = 0 
    for city in range(0,len(sample_tour)-1):
        a = sample_tour[city]
        b = sample_tour[city+1]
        cost += distance_matrix[a][b]
    return cost 

#Heuristic cost function
def heuristic(sample_tour):
    if path_cost(sample_tour) == 0:
        return 0
    return -1 * path_cost(sample_tour)

#Cooling schedule; update value of T
#In the extended implementation of the algorithm,
def cooling_schedule(t,increment): 
    def geometric_cooling():
        T = t/increment
        return T
    return geometric_cooling()

#In the extended implementation, a dynamic successor function is employed
def dynamic_successor(sample_tour):
     #Naively generate a valid shuffle of the current tour
    def naive_successor():
        next_successor = sample_tour[0:-1].copy()
        random.shuffle(next_successor)
        next_successor.append(next_successor[0])
        return next_successor
    
    #Swap of 2 adjacent cities in the tour; a randomly selected city and its right neighbor are chosen
    def swap_adjacent_successor_right():
        next_successor = sample_tour[0:-1].copy()
        i = random.randrange(len(next_successor)-1)
        next_successor[i], next_successor[i+1] = next_successor[i+1], next_successor[i]
        next_successor.append(next_successor[0])
        return next_successor
    
    #Swap of 2 adjacent cities in the tour; a randomly selected city and its left neighbor are chosen
    def swap_adjacent_successor_left():
        next_successor = sample_tour[0:-1].copy()
        i = random.randrange(len(next_successor)-1)
        next_successor[i], next_successor[i-1] = next_successor[i-1], next_successor[i]
        next_successor.append(next_successor[0])
        return next_successor
    
    #Swap 2 cities at randomly specified indices
    def swap_successor():
        next_successor = sample_tour[0:-1].copy()
        i = random.randrange(len(next_successor)-1)
        j = random.randrange(len(next_successor)-1)
        if i == j:
            swap_successor()
        next_successor[i], next_successor[j] = next_successor[j], next_successor[i]
        next_successor.append(next_successor[0])
        return next_successor      
    
    # Insert random city at specified index, shifting remaining cities to the right until original index of random city is met     
    def insert_successor():
        next_successor = sample_tour[0:-1].copy()
        i = random.randrange(len(next_successor)-1)
        j = random.randrange(len(next_successor)-1)
        while True:
            if i == j:
                insert_successor()
            if j < i:    
                i,j = j,i
            stored_j = next_successor[j]
            while j > i:
                next_successor[j] = next_successor[j-1]
                j -= 1
            next_successor[i] = stored_j
            next_successor.append(next_successor[0])
            return next_successor

    #Reverse the ordering of a random subset of the tour        
    def reverse_subset_successor():
        next_successor = sample_tour[0:-1].copy()
        i = random.randrange(len(next_successor)-1)
        j = random.randrange(len(next_successor)-1)
        while True:
            if i == j:
                reverse_subset_successor()
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
    #Evaluate the various successor states
    next_successors = [naive_successor(), swap_adjacent_successor_right(), swap_adjacent_successor_left(), swap_successor(), insert_successor(), reverse_subset_successor(), reverse_subset_successor()]
    return next_successors

#Select the successor with minimum path cost
def choose_successor(sample_tour):
    possible_successors = dynamic_successor(sample_tour)
    cost = []
    for successor in possible_successors:
        cost.append(path_cost(successor))
    min_cost = min(cost)
    return possible_successors[cost.index(min_cost)]

#Probability function; depends on values of delta E and T
def probability_calc(deltaE,T):
    calc = abs(deltaE)/T
    probability = decimal.Decimal(calc).exp()
    probability = 1/probability
    return probability

def simulated_annealing(t, max_iterations, cooling_increment, temp_breakpoint):
    #Generate an initial tour by greedy search 
    greedy_start = random.randrange(num_cities)
    initial_tour = []
    initial_tour.append(greedy_start)
    current = greedy_start
    for i in range(1,num_cities):
        cost = math.inf
        closest_city = math.inf
        for j in range(0,num_cities):
            if j not in initial_tour:
                new_cost = distance_matrix[current][j]
                if new_cost < cost:
                    cost = new_cost
                    closest_city = j
        initial_tour.append(closest_city)
        current = closest_city
    initial_tour.append(initial_tour[0])
    current = initial_tour
    T = t
    hot_break = 0.95 * T
    cold_break = 0.01 * T
    i=0
    global_min = initial_tour
    global_min_cost = path_cost(initial_tour)
    while True:
        if T > hot_break or T < cold_break:
            T = cooling_schedule(T,cooling_increment)
        else:
            T = cooling_schedule(T,temp_breakpoint)
        if i == max_iterations or T <=1:
            break
        else:
            #Randomly choose a successor of the current state
            successor = choose_successor(current)
            if path_cost(successor) < global_min_cost:
                global_min = successor
                global_min_cost = path_cost(successor)
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
        i +=1
    tour = current
    if global_min_cost < path_cost(tour):
        tour = global_min
    return tour

#Generate tour for supplied input data
tour = simulated_annealing(85,45000, 1.0003, 1.00010179)
tour_length = path_cost(tour)
    
