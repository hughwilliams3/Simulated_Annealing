import pandas as pd
from gurobipy import *
import numpy as np


def generate_knapsack_dataset(n_items=50, weight_range=(1, 100), value_range=(1, 100), capacity_ratio=0.4):
    weights = np.random.randint(weight_range[0], weight_range[1]+1, size=n_items)
    values = np.random.randint(value_range[0], value_range[1]+1, size=n_items)
    
    capacity = int(sum(weights) * capacity_ratio)
    
    df = pd.DataFrame({
        "Item": range(1, n_items+1),
        "Weight": weights,
        "Value": values
    })
    return df, capacity

def run_gurobi(df, capacity):
    m = Model("Knapsack")

    dicts = df.to_dict(orient='records')
    item_names = [a["Item"] for a in dicts]
    print(dicts)

    x = {}
    w = {}
    v = {}
    for i in item_names:
        x[i] = m.addVar(vtype = GRB.BINARY)
        
    for i in range(len(dicts)):
        w[i] = dicts[i]["Weight"]
        v[i] = dicts[i]['Value']

    m.addConstr(quicksum(w[i-1] * x[i] for i in item_names) <= capacity)

    m.setObjective(quicksum(v[i-1] * x[i] for i in item_names))

    m.ModelSense = GRB.MAXIMIZE

    m.optimize()

    binary_list = [int(round(x[i].X)) for i in item_names]
    obj_val = m.ObjVal
    return binary_list, obj_val



