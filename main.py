import pandas as pd 
import random
import math
import numpy as np


# We will be doing the rocket ship with 6 experiments

#Dataframe
file_path = '/Users/hughwilliams/Documents/Math/Operations_Research/Portfolio_Projects/Simulated_Annealing/rocket_ship.xlsx'
df = pd.read_excel(file_path)

print(df)

# initial soln will be random
# objective function will be


# initial soln
x_init = [random.randint(0, 1) for _ in range(6)]

# mu
mu = 1




def get_neighbor(x):
    bit = random.randint(0,5)
    if x[bit] == 1:
        x[bit] = 0
    else:
        x[bit] = 1
    return x

def penalty(x):
    rh = sum(df.loc[i, 'Weight']*x[i] for i in range(6)) - 220
    delta = max(0,rh)
    global mu
    pen = mu * (delta/220)
    return pen


def get_obj_val(x):
    obj_val = sum(df.loc[i, 'Merit']*x[i] for i in range(len(df))) - penalty(x)
    return obj_val

# y is the new one (x_j) and x is the old one (x_i), c is temperature
def accept(x,y,c):
    f_y = get_obj_val(y)
    f_x = get_obj_val(x)
    if f_y > f_x:
        return y
    else:
        e = math.e
        prob = e ** ((f_y - f_x)/c)
        success = np.random.binomial(n=1, p=prob)
        if success == 1:
            return y
        else:
            return x
        


print(accept([0,0,0,1,1,0],[0,0,0,1,0,0],100))
