import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from SALib.analyze import morris
from SALib.sample.morris import sample
from SALib.test_functions import Ishigami, Sobol_G
from SALib.util import read_param_file
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms

# Read the parameter range file and generate samples
problem = read_param_file('Sobol_G.txt')

# Generate samples
param_values = sample(problem, N=1000, num_levels=4, grid_jump=2, optimal_trajectories=None)

# To use optimized trajectories (brute force method), give an integer value for optimal_trajectories

# Run the "model" -- this will happen offline for external models
Y = Sobol_G.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(problem, param_values, Y, conf_level=0.95, 
                    print_to_console=True,
                    num_levels=4, grid_jump=2, num_resamples=100)

print ('Convergence index:', max(Si['mu_star_conf']/Si['mu_star']))

fig, (ax1, ax2) = plt.subplots(1, 2)
horizontal_bar_plot(ax1, Si,{}, sortby='mu_star', unit=r"tCO$_2$/year")
covariance_plot(ax2, Si, {}, unit=r"tCO$_2$/year")
fig.savefig('morris.png')

fig2 = plt.figure()
sample_histograms(fig2, param_values, problem, {'color':'y'})
fig2.savefig('morris2.png')
