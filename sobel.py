from SALib.sample import saltelli
from SALib.analyze import sobol

def ET(X):
    # column 0 = C, column 1 = R, column 2 = t
    return(0.0031*X[:,0]*(X[:,1]+209)*(X[:,2]*(X[:,2]+15))**-1)

problem = {'num_vars': 3,
           'names': ['C', 'R', 't'],
           'bounds': [[10, 100],
                     [3, 7],
                     [-10, 30]]
           }

# Generate samples
param_values = saltelli.sample(problem, 10000000, calc_second_order=False)

# Run model (example)
Y = ET(param_values)

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)
print(Si)
