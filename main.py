import pandas as pd 
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from bnb import *


# We will be doing the rocket ship with 6 experiments

def get_neighbor(x):
    neighbor = x.copy()
    bit = random.randint(0, 5)
    neighbor[bit] = 1 - neighbor[bit]
    return neighbor

def penalty(x):
    rh = sum(df.loc[i, 'Weight']*x[i] for i in range(6)) - 220
    delta = max(0,rh)
    global mu
    pen = mu * (delta/220)
    return pen

def get_obj_val(x):
    obj_val = sum(df.loc[i, 'Value']*x[i] for i in range(len(df))) - penalty(x)
    return obj_val

# y is the new one (x_j) and x is the old one (x_i), c is temperature
def accept(x,y,c):
    f_y = get_obj_val(y)
    f_x = get_obj_val(x)
    if f_y > f_x:
        print("y was better than x, accepted y")
        return y
    else:
        print("x was better than y")
        prob = math.exp((f_y - f_x)/c)
        prob = min(1, max(prob, 0))
        print(prob)
        print("\n")
        success = np.random.binomial(n=1, p=prob)
        if success == 1:
            print("y was worse but we accepted y")
            return y
        else:
           # print("y was worse and we rejected y")
            return x

def markov_chain(x,c,L_k):
    current = x
    for _ in range(L_k):
        y = get_neighbor(current)
        new_x = accept(current,y,c)
        current = new_x
    return current


if __name__ == '__main__':

    #Dataframe for rocket ship problem
    #file_path = '/Users/hughwilliams/Documents/Math/Operations_Research/Portfolio_Projects/Simulated_Annealing/rocket_ship.xlsx'
    #df = pd.read_excel(file_path)
    #C = 220

    df, C = generate_knapsack_dataset()

    # penalty parameter, mu
    mu = 1000

    # length of markov chain, L_k
    L_k = 100

    # initial temp, c0
    c0 = C

    # temperature decrement, alpha
    alpha = 0.99

    # initial soln
    x_init = [random.randint(0, 1) for _ in range(len(df))]


    c = c0
    obj_data = []
    weights_data = []
    values_data = []
    
    while True:
        if c <= c0 / L_k:
            break
        x = markov_chain(x_init,c,L_k)
        
        f_x = get_obj_val(x)
        obj_data.append(f_x)
        
        w = sum(df.loc[i, 'Weight']*x[i] for i in range(len(df)))
        weights_data.append(w)
        
        v = sum(df.loc[i, 'Value']*x[i] for i in range(len(df)))
        values_data.append(v)
        
        x_init = x
        c = alpha * c

    
    gurobi_soln, gurobi_obj_val = run_gurobi(df, C)
    print("gurobi solution:")
    print(gurobi_soln)
    print(f"gurobi objective value: {gurobi_obj_val}")
    print()
    print("simulated annealing solution:")
    print(x_init)
    print(f"simulated annealing objective value: {get_obj_val(x_init)}")


"""
    plt.plot(obj_data, '.')
    plt.title("Penalized Objective Function Evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Penalized Objective Value")
    plt.show()
    
    plt.clf()
    
    plt.plot(weights_data, 'r.', label = "Weights")
    plt.plot(values_data, 'g.', label = "Values")
    plt.title("Weight and Value Evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Weight and Value")
    plt.legend()
    plt.show()"""
    
    