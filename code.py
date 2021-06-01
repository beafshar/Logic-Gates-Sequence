import pandas as pd
import numpy as np
import time


def create_gates(idx,a,b):
    if idx == 0: return a & b 
    elif idx == 1: return a | b 
    elif idx == 2: return ~(a & b)
    elif idx == 3: return ~(a | b) 
    elif idx == 4: return a^b
    if idx == 5: return ~(a^b)

def print_results(result):
    print("the result is:")
    for x in result:
        if x == 0: print("AND", end= " ")
        elif x == 1: print("OR", end= " ")
        elif x == 2: print("NAND", end= " ")
        elif x == 3: print("NOR", end= " ")
        elif x == 4: print("XOR", end= " ")
        if x == 5: print("XNOR", end= " ")
    print()

def calc_chromosome(population,shape):
    output = create_gates(population[0],df[cols[0]],df[cols[1]])
    for i in range(1,shape):
        output = create_gates(population[i],output,df[cols[i+1]])
        
    return output
        

def calc_fitness(output,power):
    return (sum(df['Output'] == output))**power

def cross_over(x,y,p_c= 0.7):
    p = np.random.rand()
    if p > p_c:
        return x,y,False
    point = np.random.randint(low=0, high=len(x))
    x1 = x[:point]
    x2 = x[point:]
    y1 = y[:point]
    y2 = y[point:]
    child1 = np.concatenate((x1, y2), axis=None)
    child2 = np.concatenate((y1, x2), axis=None)
    return child1,child2,True
    
def mutation(x,p_m = 0.4):
    for i in range(len(x)):
        p = np.random.rand()
        if p < p_m:
            point = np.random.randint(low=0, high=6)
            x[i] = point
    return x
          



df = pd.read_csv("truth_table.csv")
cols = list(df.columns)

genes = len(cols) - 2
print("Calculating...")


chromosomes = 2**(genes)
pop_size = (chromosomes,genes)
start = time.time()
new_population_idx = np.random.randint(low=0, high=6, size=pop_size)

num_of_generations = 1
max_fit = 0
n = 0
power = 4
steps = 4

while(True):
    new_population = np.array([calc_chromosome(pop_idx,genes) for pop_idx in new_population_idx])
    fit = []
    fit = np.array([calc_fitness(pop,power) for pop in new_population])
    if max(fit) == max_fit:
        n += 1
    else:
        n = 0
    
    if n < steps:
        p_m = 0.1
    else:
        p_m = 0.4
        n = 0
    max_fit = max(fit)

    if (df.shape[0])**power == max(fit):
        print("Achieved goal in " + str(num_of_generations) +
              " generations and in " + str(time.time() - start) + " seconds.")
        print_results(new_population_idx[np.argmax(fit)])
        
        break
    
    fitness = fit/sum(fit)

    parrents_num = len(new_population)
    indices = np.random.choice(np.arange(len(fitness)),replace = True, size =parrents_num,p = fitness)

    children = np.array([[0]* genes])
    parrents = np.array([0])
    
    
    if max(fit) < ((0.9 * df.shape[0])**power):
        p_m = 0.4
        p_c = 0.85
    else:
        p_m = 0.1
        p_c = 0.7

    for i in range(0,len(indices),2):
        par_idx1 = indices[i]
        par_idx2 = indices[i+1]

        parrent1, parrent2 = new_population_idx[par_idx1],new_population_idx[par_idx2]
        child1,child2,flag = cross_over(parrent1,parrent2,p_c = p_c)
            
        child1 = mutation(child1,p_m)
        child2 = mutation(child2,p_m)
        children = np.append(children,[child1,child2],axis= 0)


    children = np.delete(children,0,axis = 0)
    new_population_idx = children
    
    num_of_generations += 1
   