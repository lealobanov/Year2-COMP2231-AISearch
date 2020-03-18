import os
import sys
import time
import random
import decimal
import math

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

alg_code = "SA"

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
    
    











    


