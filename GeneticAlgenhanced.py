import os
import sys
import time
import random
import math

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
    for city in range(0,len(sample_tour)-1):
        a = sample_tour[city]
        b = sample_tour[city+1]
        cost += distance_matrix[a][b]
    return cost 

#Define fitness function (aim to maximize); treat fitness as inverse of tour cost
def fitness(population):
    fitnesses = []
    for tour in population:
        check_path_cost = path_cost(tour)
        if check_path_cost == 0:
            tour_fitness = 101
        else:
            tour_fitness = 100*(1/check_path_cost)
        fitnesses.append(tour_fitness)
    return fitnesses

def fittest_individual(population):
    #Find and store the fittest individual
    fitnesses = fitness(population)
    sorted_fitnesses = sorted(fitnesses,reverse=True)
    fittest_member = sorted_fitnesses[0]
    best_tour_index = fitnesses.index(fittest_member)
    best_tour = population[best_tour_index]
    return best_tour

#Select 2 parents from the population to breed; fitter individuals are more likely to be selected 
#In the enhanced implementation, k-tournament selection is used
def tournament_selection(mating_pool, k):
    #Select k random members from the mating pool to participate in tournament
    k_chosen = []
    for i in range(0,k):
        chosen_position = random.randint(0,len(mating_pool)-1)
        chosen_member = mating_pool[chosen_position]
        k_chosen.append(chosen_member)
    #From these k members, select the fittest to be a parent
    fittest_in_k = fittest_individual(k_chosen)
    return fittest_in_k
    

#Crossover parents a and b with probability crossover_rate
def crossover(a,b, crossover_rate):
    #Ordered crossover
    def ordered_crossover():
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
    def select_crossover():
        select_random = random.random()
        #Check if crossover rate is met; if yes, proceed to implement ordered crossover and produce a child
        if crossover_rate >= select_random:
            selected_child = ordered_crossover()
            return selected_child
        #If the crossover rate is not met, return the second parent (the first parent is the fittest member of the current population; hence, automatically part of the next generation due to use of elitism)
        else:
            return b
    return select_crossover()
    
#In the extended implementation, various mutation operators are employed; if the mutation probability is met, the fittest of the produced mutations is selected as the mutated child
#Mutation operators attributed to: https://www.researchgate.net/publication/304623320_Enhancing_genetic_algorithms_using_multi_mutations
def mutation(child, probability):
    mut_child = child[0:-1].copy()
    #Perform various pre-processing operations necessary for mutation operators
    #
    #Find the worst gene in the child tour 
    #The worst gene is defined as city with max distance to its left neighbor; for the start city, distance to end city (index -1) is used
    worst_score = distance_matrix[0][0]
    worst_gene = 0
    for i in range(1,len(mut_child)):
        distance = distance_matrix[i][i-1]
        if distance > worst_score:
            worst_score = distance
            worst_gene = i

    #Find the worst LR gene in the child tour
    #Find worst gene wrt left and right neighbors
    worst_LRscore = distance_matrix[0][0]+distance_matrix[0][1]
    worst_LRgene = 0
    for i in range(1,len(mut_child)-1):
        distance = distance_matrix[i][i-1] + distance_matrix[i][i+1]
        if distance > worst_LRscore:
            worst_LRscore = distance
            worst_LRgene = i
    last_distance = distance_matrix[len(mut_child)-1][len(mut_child)-2] + distance_matrix[len(mut_child)-1][len(mut_child)-1]
    if last_distance > worst_LRscore:
            worst_LRscore = last_distance
            worst_LRgene = len(mut_child)-1

   #Find the nearest neighbor wrt to the worst LR gene
    nearest = math.inf
    nearest_index = math.inf
    for i in range(0,num_cities):
        score = distance_matrix[worst_LRgene][i]
        if score < nearest:
            nearest = score
            nearest_index = mut_child.index(i)

    #Specify a search range to use in mutation operators
    search_range = 5

    #Swap 2 cities at randomly specified indices
    def random_swap():
        mutating = mut_child.copy()
        i = random.randrange(len(mut_child))
        j = random.randrange(len(mut_child))
        if i == j:
            random_swap()
        mutating[i], mutating[j] = mutating[j], mutating[i]
        mutating.append(mutating[0]) 
        return mutating

    #Swap the worst gene with a randomly selected city
    def worst_gene_random_gene():
        mutating = mut_child.copy()
        i = random.randrange(len(mutating))
        mutating[i], mutating[worst_gene] = mutating[worst_gene], mutating[i]
        mutating.append(mutating[0]) 
        return mutating

    #Swap the two worst genes 
    def two_worst_genes():
        mutating = mut_child.copy()
        second_worst_score = distance_matrix[0][0]
        second_worst_gene = 0
        for i in range(1,len(mut_child)):
            distance = distance_matrix[i][i-1]
            if distance > second_worst_score and distance <= worst_score:
                second_worst_score = distance
                second_worst_gene = i
        mutating[second_worst_gene], mutating[worst_gene] = mutating[worst_gene], mutating[second_worst_gene]
        mutating.append(mutating[0]) 
        return mutating 

    #Swap the worst LR city with a random city
    def worst_LR_gene_random():
        mutating = mut_child.copy()
        i = random.randrange(len(mut_child))
        mutating[worst_LRgene], mutating[i] = mutating[i], mutating[worst_LRgene]
        mutating.append(mutating[0]) 
        return mutating 

    #Swap the worst LR city with its nearest neighbor to the right
    def worst_gene_nearest_neighbor():
        mutating = mut_child.copy()
        #Within the search range, find the index of the nearest city to the worst LR city
        i = random.randrange(search_range+1) + nearest_index
        try: 
            swap = mutating[i]
        except IndexError:
            i = i%num_cities
        mutating[worst_LRgene], mutating[i] = mutating[i], mutating[worst_LRgene]
        mutating.append(mutating[0]) 
        return mutating

    #Swap the worst LR city with its nearest neighbor to the right and left
    def worst_gene_worst_nearest_neighbor():
        mutating = mut_child.copy()
        #Within the search range, find the index of the nearest city to the worst LR city
        worstNN_score = 0 
        worstNN_gene = 0 
        #Check to the right
        for i in range(1,search_range):
            try: 
                swap_right = mutating[nearest_index+i]
                distance = distance_matrix[nearest_index][swap_right]
                if distance > worstNN_score:
                    worstNN_score = distance
                    worstNN_gene = nearest_index+i
            #Wrap around indices
            except IndexError:
                j = (nearest_index+i)%num_cities
                swap_right = mutating[j]
                distance = distance_matrix[nearest_index][swap_right]
                if distance > worstNN_score:
                    worstNN_score = distance
                    worstNN_gene = j
        #Check to the left
        for i in range(1,search_range):
            try: 
                swap_left = mutating[nearest_index-i]
                distance = distance_matrix[nearest_index][swap_left]
                if distance > worstNN_score:
                    worstNN_score = distance
                    worstNN_gene = nearest_index-i
            #Wrap around indices 
            except IndexError:
                j = -1*((nearest_index-i)%num_cities)
                swap_left = mutating[j]
                distance = distance_matrix[nearest_index][swap_left]
                if distance > worstNN_score:
                    worstNN_score = distance
                    worstNN_gene = j
        mutating[worst_LRgene], mutating[worstNN_gene] = mutating[worstNN_gene], mutating[worst_LRgene]
        mutating.append(mutating[0]) 
        return mutating  

    #Swap genes local to the worst gene; return the fittest mutated child    
    def worst_gene_local_swap():
        mut_childA = mut_child.copy()
        mut_childB = mut_child.copy()
        #Swap the two elements to the left of the worst gene
        mut_childA[worst_gene-1], mut_childA[worst_gene-2] = mut_childA[worst_gene-2], mut_childA[worst_gene-1]
        #Swap the worst gene with the element to its right
        mut_childB[worst_gene], mut_childB[worst_gene+1] = mut_childB[worst_gene+1], mut_childB[worst_gene]
        #Compare the fitness of the two tours; return the best as the mutation
        if path_cost(mut_childA) <= path_cost(mut_childB):
            mut_childA.append(mut_childA[0])
            return mut_childA
        else:
            mut_childB.append(mut_childB[0])
            return mut_childB

    #Reverse the ordering of a random subset of the tour
    def reverse_subset_successor():
        mutating = mut_child.copy()
        i = random.randrange(len(mutating))
        j = random.randrange(len(mutating))
        while True:
            if i == j:
                reverse_subset_successor()
            if i < j:    
                subset = mutating[i:j+1]
                z=i
            else: 
                subset = mutating[j:i+1]
                z=j
            subset.reverse()
            for entry in subset:
                mutating[z] = entry
                z+=1
            mutating.append(mutating[0])
            return mutating

    #Naively generate a valid shuffle of the current tour
    def naive_shuffle():
        mutating = mut_child.copy()
        random.shuffle(mutating)
        mutating.append(mutating[0])
        return mutating

    #Apply the various mutation operators to the child tour; if mutation probability is met, return the fittest mutated tour      
    def select_best_mutation(child):
        select_random = random.random()
        if probability >= select_random:
            #Some mutation operators are called multiple times 
            mut_child_options = [random_swap(), random_swap(),worst_gene_random_gene(), worst_gene_random_gene(), two_worst_genes(), worst_LR_gene_random(), worst_LR_gene_random(), worst_gene_worst_nearest_neighbor(), worst_gene_local_swap(), worst_gene_local_swap(), reverse_subset_successor(), reverse_subset_successor(), naive_shuffle()]
            tour_cost = math.inf
            for tour in mut_child_options:
                if path_cost(tour) < tour_cost:
                    best_mutation = tour
                    tour_cost = path_cost(tour)
            return best_mutation
        else:
            return child
    return select_best_mutation(child)

#Produce the next generation
#In the extended implementation, elitism and crossover rate is introduced
def next_generation(current_generation, elitism, crossover_rate, mutation_rate):
    next_gen = []
    current_gen_fitness = fitness(current_generation)
    sorted_fitnesses = sorted(current_gen_fitness,reverse=True)
    pop_size = len(current_generation)
    fitq = fittest_individual(current_generation)
    #Check to see if early convergence has occurred (if all members of the current population are equal); if yes, force the mutation probability to 1 when producing the next generation
    q = 0
    for city in current_generation:
        if city != fitq:
            q = 1
    if q == 0:
        mutation_rate = 1
    #Implement elitism; best x% of the current population automatically carry over to the subsequent generation
    automatic_carry_over = math.floor(elitism * pop_size)
    for i in range(0,automatic_carry_over):
        add_member = sorted_fitnesses[i]
        member_index = current_gen_fitness.index(add_member)
        add_member = current_generation[member_index]
        next_gen.append(add_member)
    #In the extended implementation, a mating pool is used; only the x% fittest individuals will mate to produce offspring 
    mating_pool = []
    #After experimentation, the proportion of individuals to mate (participate in parent selection) has been set to 0.5
    mating_proportion = 0.5
    mating_pool_size = math.floor(mating_proportion * pop_size)
    i=0
    while len(mating_pool) != mating_pool_size:
        add_member = sorted_fitnesses[i]  
        member_index = current_gen_fitness.index(add_member)
        add_member = current_generation[member_index] 
        mating_pool.append(add_member)
        i+=1
    i = 0 
    while i < len(current_generation)-automatic_carry_over:
        #In the extended implementation, parent selection has been modified - only 1 parent is selected
        #The second parent is automatically set as the fittest member of the current generation
        child = mutation(crossover(fitq, tournament_selection(mating_pool,3), crossover_rate), mutation_rate)
        next_gen.append(child[:])
        i +=1
    return next_gen

def genetic_algorithm(pop_size, generations, elitism, crossover_rate, mutation_rate):
    #Create an initial population
    current_population = population(pop_size)
    #Identify an initial best tour; used as a basis of comparison for all future tours generated
    best_tour = fittest_individual(current_population)
    best_tour_cost = path_cost(best_tour)
    #Produce subsequent generations 
    for i in range(0, generations):
        current_population = next_generation(current_population, elitism, crossover_rate, mutation_rate)
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
full_tour = genetic_algorithm(15,1500,0.1,0.7,0.07)
tour_length = path_cost(full_tour)
tour = full_tour[:-1]

