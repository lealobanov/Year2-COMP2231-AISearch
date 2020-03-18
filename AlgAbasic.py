import os
import sys
import time
import random

def read_file_into_string(input_file, from_ord, to_ord):
    # take a file "input_file", read it character by character, strip away all unwanted
    # characters with ord < "from_ord" and ord > "to_ord" and return the concatenation
    # of the file as the string "output_string"
    the_file = open(input_file,'r')
    current_char = the_file.read(1)
    output_string = ""
    while current_char != "":
        if ord(current_char) >= from_ord and ord(current_char) <= to_ord:
            output_string = output_string + current_char
        current_char = the_file.read(1)
    the_file.close()
    return output_string

def stripped_string_to_int(a_string):
    # take a string "a_string" and strip away all non-numeric characters to obtain the string
    # "stripped_string" which is then converted to an integer with this integer returned
    a_string_length = len(a_string)
    stripped_string = "0"
    if a_string_length != 0:
        for i in range(0,a_string_length):
            if ord(a_string[i]) >= 48 and ord(a_string[i]) <= 57:
                stripped_string = stripped_string + a_string[i]
    resulting_int = int(stripped_string)
    return resulting_int

def get_string_between(from_string, to_string, a_string, from_index):
    # look for the first occurrence of "from_string" in "a_string" starting at the index
    # "from_index", and from the end of this occurrence of "from_string", look for the first
    # occurrence of the string "to_string"; set "middle_string" to be the sub-string of "a_string"
    # lying between these two occurrences and "to_index" to be the index immediately after the last
    # character of the occurrence of "to_string" and return both "middle_string" and "to_index"
    middle_string = ""              # "middle_string" and "to_index" play no role in the case of error
    to_index = -1                   # but need to initialized to something as they are returned
    start = a_string.find(from_string,from_index)
    if start == -1:
        flag = "*** error: " + from_string + " doesn't appear"
        #trace_file.write(flag + "\n")
    else:
        start = start + len(from_string)
        end = a_string.find(to_string,start)
        if end == -1:
            flag = "*** error: " + to_string + " doesn't appear"
            #trace_file.write(flag + "\n")
        else:
            middle_string = a_string[start:end]
            to_index = end + len(to_string)
            flag = "good"
    return middle_string,to_index,flag

def string_to_array(a_string, from_index, num_cities):
    # convert the numbers separated by commas in the file-as-a-string "a_string", starting from index "from_index",
    # which should point to the first comma before the first digit, into a two-dimensional array "distances[][]"
    # and return it; note that we have added a comma to "a_string" so as to find the final distance
    # distance_matrix = []
    if from_index >= len(a_string):
        flag = "*** error: the input file doesn't have any city distances"
        #trace_file.write(flag + "\n")
    else:
        row = 0
        column = 1
        row_of_distances = [0]
        flag = "good"
        while flag == "good":
            middle_string, from_index, flag = get_string_between(",", ",", a_string, from_index)
            from_index = from_index - 1         # need to look again for the comma just found
            if flag != "good":
                flag = "*** error: there aren't enough cities"
                # trace_file.write(flag + "\n")
            else:
                distance = stripped_string_to_int(middle_string)
                row_of_distances.append(distance)
                column = column + 1
                if column == num_cities:
                    distance_matrix.append(row_of_distances)
                    row = row + 1
                    if row == num_cities - 1:
                        flag = "finished"
                        row_of_distances = [0]
                        for i in range(0, num_cities - 1):
                            row_of_distances.append(0)
                        distance_matrix.append(row_of_distances)
                    else:
                        row_of_distances = [0]
                        for i in range(0,row):
                            row_of_distances.append(0)
                        column = row + 1
        if flag == "finished":
            flag = "good"
    return flag

def make_distance_matrix_symmetric(num_cities):
    # make the upper triangular matrix "distance_matrix" symmetric;
    # note that there is nothing returned
    for i in range(1,num_cities):
        for j in range(0,i):
            distance_matrix[i][j] = distance_matrix[j][i]

# read input file into string

#######################################################################################################
############ now we read an input file to obtain the number of cities, "num_cities", and a ############
############ symmetric two-dimensional list, "distance_matrix", of city-to-city distances. ############
############ the default input file is given here if none is supplied via a command line   ############
############ execution; it should reside in a folder called "city-files" whether it is     ############
############ supplied internally as the default file or via a command line execution.      ############
############ if your input file does not exist then the program will crash.                ############

input_file = "AISearchfile175.txt"

#######################################################################################################

# you need to worry about the code below until I tell you; that is, do not touch it!

if len(sys.argv) == 1:
    file_string = read_file_into_string("../city-files/" + input_file,44,122)
else:
    input_file = sys.argv[1]
    file_string = read_file_into_string("../city-files/" + input_file,44,122)
file_string = file_string + ","         # we need to add a final comma to find the city distances
                                        # as we look for numbers between commas
print("I'm working with the file " + input_file + ".")
                                        
# get the name of the file

name_of_file,to_index,flag = get_string_between("NAME=", ",", file_string, 0)

if flag == "good":
    print("I have successfully read " + input_file + ".")
    # get the number of cities
    num_cities_string,to_index,flag = get_string_between("SIZE=", ",", file_string, to_index)
    num_cities = stripped_string_to_int(num_cities_string)
else:
    print("***** ERROR: something went wrong when reading " + input_file + ".")
if flag == "good":
    print("There are " + str(num_cities) + " cities.")
    # convert the list of distances into a 2-D array
    distance_matrix = []
    to_index = to_index - 1             # ensure "to_index" points to the comma before the first digit
    flag = string_to_array(file_string, to_index, num_cities)
if flag == "good":
    # if the conversion went well then make the distance matrix symmetric
    make_distance_matrix_symmetric(num_cities)
    print("I have successfully built a symmetric two-dimensional array of city distances.")
else:
    print("***** ERROR: something went wrong when building the two-dimensional array of city distances.")

#######################################################################################################
############ end of code to build the distance matrix from the input file: so now you have ############
############ the two-dimensional "num_cities" x "num_cities" symmetric distance matrix     ############
############ "distance_matrix[][]" where "num_cities" is the number of cities              ############
#######################################################################################################

# now you need to supply some parameters ...

#######################################################################################################
############ YOU NEED TO INCLUDE THE FOLLOWING PARAMETERS:                                 ############
############ "my_user_name" = your user-name, e.g., mine is dcs0ias                        ############

my_user_name = "cssg28"

############ "my_first_name" = your first name, e.g., mine is Iain                         ############

my_first_name = "Lea"

############ "my_last_name" = your last name, e.g., mine is Stewart                        ############

my_last_name = "Lobanov"

############ "alg_code" = the two-digit code that tells me which algorithm you have        ############
############ implemented (see the assignment pdf), where the codes are:                    ############
############    BF = brute-force search                                                    ############
############    BG = basic greedy search                                                   ############
############    BS = best_first search without heuristic data                              ############
############    ID = iterative deepening search                                            ############
############    BH = best_first search with heuristic data                                 ############
############    AS = A* search                                                             ############
############    HC = hilling climbing search                                               ############
############    SA = simulated annealing search                                            ############
############    GA = genetic algorithm                                                     ############

alg_code = "GA"

############ you can also add a note that will be added to the end of the output file if   ############
############ you like, e.g., "in my basic greedy search, I broke ties by always visiting   ############
############ the first nearest city found" or leave it empty if you wish                   ############

added_note = ""

############ the line below sets up a dictionary of codes and search names (you need do    ############
############ nothing unless you implement an alternative algorithm and I give you a code   ############
############ for it when you can add the code and the algorithm to the dictionary)         ############

codes_and_names = {'BF' : 'brute-force search',
                   'BG' : 'basic greedy search',
                   'BS' : 'best_first search without heuristic data',
                   'ID' : 'iterative deepening search',
                   'BH' : 'best_first search with heuristic data',
                   'AS' : 'A* search',
                   'HC' : 'hilling climbing search',
                   'SA' : 'simulated annealing search',
                   'GA' : 'genetic algorithm'}

#######################################################################################################
############    now the code for your algorithm should begin                               ############
#######################################################################################################

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















#######################################################################################################
############ the code for your algorithm should now be complete and you should have        ############
############ computed a tour held in the list "tour" of length "tour_length"               ############
#######################################################################################################

# you do not need to worry about the code below; that is, do not touch it

#######################################################################################################
############ start of code to verify that the constructed tour and its length are valid    ############
#######################################################################################################

check_tour_length = 0
for i in range(0,num_cities-1):
    check_tour_length = check_tour_length + distance_matrix[tour[i]][tour[i+1]]
check_tour_length = check_tour_length + distance_matrix[tour[num_cities-1]][tour[0]]
flag = "good"
if tour_length != check_tour_length:
    flag = "bad"
if flag == "good":
    print("Great! Your tour-length of " + str(tour_length) + " from your " + codes_and_names[alg_code] + " is valid!")
else:
    print("***** ERROR: Your claimed tour-length of " + str(tour_length) + "is different from the true tour length of " + str(check_tour_length) + ".")

#######################################################################################################
############ start of code to write a valid tour to a text (.txt) file of the correct      ############
############ format; if your tour is not valid then you get an error message on the        ############
############ standard output and the tour is not written to a file                         ############
############                                                                               ############
############ the name of file is "my_user_name" + mon-dat-hr-min-sec (11 characters);      ############
############ for example, dcs0iasSep22105857.txt; if dcs0iasSep22105857.txt already exists ############
############ then it is overwritten                                                        ############
#######################################################################################################

if flag == "good":
    local_time = time.asctime(time.localtime(time.time()))   # return 24-character string in form "Tue Jan 13 10:17:09 2009"
    output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
                                                             # output_file_time = mon + day + hour + min + sec (11 characters)
    output_file_name = my_user_name + output_file_time + ".txt"
    f = open(output_file_name,'w')
    f.write("USER = " + my_user_name + " (" + my_first_name + " " + my_last_name + ")\n")
    f.write("ALGORITHM = " + alg_code + ", FILENAME = " + name_of_file + "\n")
    f.write("NUMBER OF CITIES = " + str(num_cities) + ", TOUR LENGTH = " + str(tour_length) + "\n")
    f.write(str(tour[0]))
    for i in range(1,num_cities):
        f.write("," + str(tour[i]))
    if added_note != "":
        f.write("\nNOTE = " + added_note)
    f.close()
    print("I have successfully written the tour to the output file " + output_file_name + ".")
    
    











    


