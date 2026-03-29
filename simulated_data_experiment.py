from environments import OptimismTrapInstance, FrequentSwitcherInstance, UniformRandomInstance
from core import simulate_and_plot_LCB_vs_UCB, evaluate_sensitivity_to_upsilon

T = 5000
n_seeds = 20

# The first expert will always be the best
simulate_and_plot_LCB_vs_UCB(OptimismTrapInstance, T, n_seeds, instance_name='first-best_instance')

# New expert arriving has its expected reward sampled from a uniform in [0,1]
simulate_and_plot_LCB_vs_UCB(UniformRandomInstance, T, n_seeds, instance_name='uniform_experts_instance')

# High number of optimum switches
simulate_and_plot_LCB_vs_UCB(FrequentSwitcherInstance, T, n_seeds, env_params=[int(T**(1/3))], instance_name='frequent_switches_instance')