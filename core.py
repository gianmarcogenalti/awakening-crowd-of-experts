import numpy as np
import matplotlib.pyplot as plt
from algorithms import LCBAgent, UCBAgent
from environments import FrequentSwitcherInstance, ReproducibleMarketInstance

def evaluate_agent_stochastic(agent, env, T):
    cumulative_regret = np.zeros(T)
    current_regret_sum = 0.0
    
    # Pre-allocate the array to exactly size T to prevent shape mismatches
    chosen_experts = np.zeros(T, dtype=int) 
    
    for t in range(T):
        chosen_expert = agent.pull_arm()
        
        # Save the expert chosen at this exact round
        chosen_experts[t] = chosen_expert
        
        X = env.step()
        
        # Calculate expected ordering regret against the best ordering
        optimal_mean = env.get_optimal_reward()
        agent_mean = env.true_means[chosen_expert] 
        
        current_regret_sum += (optimal_mean - agent_mean)
        cumulative_regret[t] = current_regret_sum
        
        agent.update(X)
        
    return cumulative_regret, chosen_experts

def get_final_stats(data_array, n_seeds):
    # Extract the final column (t = T-1) for all seeds
    final_values = data_array[:, -1]
    mean_val = np.mean(final_values)
    se_val = np.std(final_values) / np.sqrt(n_seeds)
    return f"{mean_val:.2f} ± {se_val:.2f}"

def simulate_and_plot_LCB_vs_UCB(env_type, T, n_seeds, instance_name=" ", env_params=[]):
    all_cum_regrets_lcb, all_switches_lcb, all_opt_pulls_lcb = [], [], []
    all_cum_regrets_ucb, all_switches_ucb, all_opt_pulls_ucb = [], [], []

    for seed in np.arange(n_seeds):
        if seed % 5 == 0:
            print(f'Seed {seed}')
            
        # --- 1. Evaluate LCB ---
        np.random.seed(seed) 
        agent_lcb = LCBAgent(T) 
        env_lcb = env_type(T, *env_params)
        regret_lcb, chosen_experts_lcb = evaluate_agent_stochastic(agent_lcb, env_lcb, T)
        
        # Force to numpy array of ints to prevent indexing errors
        chosen_experts_lcb = np.array(chosen_experts_lcb, dtype=int)
        
        # Vectorized Switches
        switches_lcb = np.concatenate(([0], np.cumsum(chosen_experts_lcb[1:] != chosen_experts_lcb[:-1])))
        
        # Vectorized Optimal Pulls (No list comprehension needed!)
        agent_means_lcb = env_lcb.true_means[chosen_experts_lcb]
        is_opt_lcb = (agent_means_lcb == env_lcb.optimal_means)
        opt_pulls_lcb = np.cumsum(is_opt_lcb)
        
        all_cum_regrets_lcb.append(regret_lcb)
        all_switches_lcb.append(switches_lcb)
        all_opt_pulls_lcb.append(opt_pulls_lcb)

        # --- 2. Evaluate UCB ---
        np.random.seed(seed) # Crucial: Reset seed for UCB
        agent_ucb = UCBAgent(T)
        env_ucb = env_type(T, *env_params)
        regret_ucb, chosen_experts_ucb = evaluate_agent_stochastic(agent_ucb, env_ucb, T)
        
        chosen_experts_ucb = np.array(chosen_experts_ucb, dtype=int)
        
        switches_ucb = np.concatenate(([0], np.cumsum(chosen_experts_ucb[1:] != chosen_experts_ucb[:-1])))
        
        agent_means_ucb = env_ucb.true_means[chosen_experts_ucb]
        is_opt_ucb = (agent_means_ucb == env_ucb.optimal_means)
        opt_pulls_ucb = np.cumsum(is_opt_ucb)
        
        all_cum_regrets_ucb.append(regret_ucb)
        all_switches_ucb.append(switches_ucb)
        all_opt_pulls_ucb.append(opt_pulls_ucb)

    # Convert all to numpy arrays at the end for your plotting code
    metrics = {
        'regret': (np.array(all_cum_regrets_lcb), np.array(all_cum_regrets_ucb)),
        'switches': (np.array(all_switches_lcb), np.array(all_switches_ucb)),
        'opt_pulls': (np.array(all_opt_pulls_lcb), np.array(all_opt_pulls_ucb))
    }


    lcb_regret = get_final_stats(metrics['regret'][0], n_seeds)
    ucb_regret = get_final_stats(metrics['regret'][1], n_seeds)
    
    lcb_switches = get_final_stats(metrics['switches'][0], n_seeds)
    ucb_switches = get_final_stats(metrics['switches'][1], n_seeds)
    
    lcb_opt_pulls = get_final_stats(metrics['opt_pulls'][0], n_seeds)
    ucb_opt_pulls = get_final_stats(metrics['opt_pulls'][1], n_seeds)
    
    # Format as a Markdown Table
    md = f"### Summary of Results: {instance_name} (T={T}, Seeds={n_seeds})\n\n"
    md += "| Metric | $\\pi_{LCB}$ (Proposed) | $\\pi_{UCB}$ (Baseline) |\n"
    md += "| :--- | :--- | :--- |\n"
    md += f"| **Final Cumulative Regret** | {lcb_regret} | {ucb_regret} |\n"
    md += f"| **Total Arm Switches** | {lcb_switches} | {ucb_switches} |\n"
    md += f"| **Total Optimal Pulls** | {lcb_opt_pulls} | {ucb_opt_pulls} |\n"
    print(md)

        # ==========================================
    # GENERATE THE 1x3 PLOT GRID
    # ==========================================
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Stochastic ACE: LCB vs UCB on {instance_name}', fontsize=16)

    titles = ['Cumulative Expected Regret', 'Cumulative Arm Switches']
    y_labels = ['Regret', 'Number of Switches']
    metric_keys = ['regret', 'switches']

    # 2. PLOT METRICS

    for i, key in enumerate(metric_keys):
        data_lcb, data_ucb = metrics[key]
        
        avg_lcb = data_lcb.mean(axis=0)
        err_lcb = data_lcb.std(axis=0) / np.sqrt(n_seeds)
        
        avg_ucb = data_ucb.mean(axis=0)
        err_ucb = data_ucb.std(axis=0) / np.sqrt(n_seeds)
        
        axs[i].plot(avg_lcb, label='LCB', color='blue', linewidth=2)
        axs[i].fill_between(np.arange(T), avg_lcb - err_lcb, avg_lcb + err_lcb, color='blue', alpha=0.2)
        
        axs[i].plot(avg_ucb, label='UCB', color='red', linewidth=2, linestyle='--')
        axs[i].fill_between(np.arange(T), avg_ucb - err_ucb, avg_ucb + err_ucb, color='red', alpha=0.2)
        
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Rounds (t)')
        axs[i].set_ylabel(y_labels[i])
        axs[i].legend(loc='upper left')
        axs[i].grid(True, alpha=0.3)

    # Add log scale to the switches plot so the LCB steps are visible
    axs[1].set_yscale('symlog')
    axs[1].set_yticks([0, 10, 100, 1000, T])
    axs[1].get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # 3. PLOT TRAJECTORY
    optimal_trajectory = env_lcb.optimal_means 
    axs[2].plot(range(T), optimal_trajectory, color='green', linewidth=2, label="Optimal Expert's Mean")
    axs[2].set_title('Example Trajectory of the Optimal Expert')
    axs[2].set_xlabel('Rounds (t)')
    axs[2].set_ylabel('True Expected Reward (μ)')
    axs[2].legend(loc='lower right')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    pdf_filename = f"plots/{instance_name}.png"
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"Plot saved successfully as: {pdf_filename}")

def evaluate_sensitivity_to_upsilon(T, n_seeds):
    # Define the experiment parameters
    # Choose the values of Upsilon* to test
    # We include small values up to values much larger than T^(1/3) (~12.6)
    upsilon_list = [np.log(T), int(T**(1/3)), int(np.sqrt(T)), int(T**(2/3)), int(T/4), int(T/2)]

    # Dictionaries to store the final means and standard errors for plotting
    results = {
        'lcb_regret_mean': [], 'lcb_regret_se': [],
        'ucb_regret_mean': [], 'ucb_regret_se': [],
        'lcb_switches_mean': [], 'lcb_switches_se': [],
        'ucb_switches_mean': [], 'ucb_switches_se': []
    }

    print(f"Starting Sensitivity Analysis for Upsilon* over {n_seeds} seeds...")

    for u_star in upsilon_list:
        print(f"Evaluating Upsilon* = {u_star}...")
        
        final_regrets_lcb, final_switches_lcb = [], []
        final_regrets_ucb, final_switches_ucb = [], []
        
        for seed in range(n_seeds):
            # --- 1. Evaluate LCB ---
            np.random.seed(seed)
            agent_lcb = LCBAgent(T)
            env_lcb = FrequentSwitcherInstance(T, upsilon_star=u_star, mean_lower=0.1, mean_upper=0.9)
            regret_lcb, chosen_experts_lcb = evaluate_agent_stochastic(agent_lcb, env_lcb, T)
            
            chosen_experts_lcb = np.array(chosen_experts_lcb, dtype=int)
            total_switches_lcb = np.sum(chosen_experts_lcb[1:] != chosen_experts_lcb[:-1])
            
            final_regrets_lcb.append(regret_lcb[-1])
            final_switches_lcb.append(total_switches_lcb)

            # --- 2. Evaluate UCB ---
            np.random.seed(seed) # Reset seed for exact same environment
            agent_ucb = UCBAgent(T)
            env_ucb = FrequentSwitcherInstance(T, upsilon_star=u_star, mean_lower=0.1, mean_upper=0.9)
            regret_ucb, chosen_experts_ucb = evaluate_agent_stochastic(agent_ucb, env_ucb, T)
            
            chosen_experts_ucb = np.array(chosen_experts_ucb, dtype=int)
            total_switches_ucb = np.sum(chosen_experts_ucb[1:] != chosen_experts_ucb[:-1])
            
            final_regrets_ucb.append(regret_ucb[-1])
            final_switches_ucb.append(total_switches_ucb)
            
        # Calculate and store statistics for this Upsilon*
        results['lcb_regret_mean'].append(np.mean(final_regrets_lcb))
        results['lcb_regret_se'].append(np.std(final_regrets_lcb) / np.sqrt(n_seeds))
        
        results['ucb_regret_mean'].append(np.mean(final_regrets_ucb))
        results['ucb_regret_se'].append(np.std(final_regrets_ucb) / np.sqrt(n_seeds))
        
        results['lcb_switches_mean'].append(np.mean(final_switches_lcb))
        results['lcb_switches_se'].append(np.std(final_switches_lcb) / np.sqrt(n_seeds))
        
        results['ucb_switches_mean'].append(np.mean(final_switches_ucb))
        results['ucb_switches_se'].append(np.std(final_switches_ucb) / np.sqrt(n_seeds))

    print("Simulation complete! Generating plots...")

    # ==========================================
    # PLOTTING THE SENSITIVITY ANALYSIS
    # ==========================================
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(r'Sensitivity to $\Upsilon^*$ (Optimal Expert Switches)', fontsize=16)

    # --- Plot 1: Final Cumulative Regret vs Upsilon* ---
    axs[0].errorbar(upsilon_list, results['lcb_regret_mean'], yerr=results['lcb_regret_se'], 
                    fmt='-o', color='blue', label='LCB', linewidth=2, capsize=5)
    axs[0].errorbar(upsilon_list, results['ucb_regret_mean'], yerr=results['ucb_regret_se'], 
                    fmt='--s', color='red', label='UCB', linewidth=2, capsize=5)

    axs[0].set_title('Final Cumulative Regret at $t=T$')
    axs[0].set_xlabel(r'Number of Optimal Switches ($\Upsilon^*$)')
    axs[0].set_ylabel('Cumulative Regret')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # --- Plot 2: Final Number of Switches vs Upsilon* ---
    axs[1].errorbar(upsilon_list, results['lcb_switches_mean'], yerr=results['lcb_switches_se'], 
                    fmt='-o', color='blue', label='LCB', linewidth=2, capsize=5)
    axs[1].errorbar(upsilon_list, results['ucb_switches_mean'], yerr=results['ucb_switches_se'], 
                    fmt='--s', color='red', label='UCB', linewidth=2, capsize=5)

    # Add a dashed line representing the actual number of optimal switches in the environment
    axs[1].plot(upsilon_list, upsilon_list, color='green', linestyle=':', linewidth=2, label='True Environment Switches')

    axs[1].set_title('Total Arm Switches by Algorithm')
    axs[1].set_xlabel(r'Number of Optimal Switches ($\Upsilon^*$)')
    axs[1].set_ylabel('Number of Algorithm Switches')
    axs[1].set_yscale('symlog') # Keep log scale to handle UCB's massive switching rate
    axs[1].get_yaxis().set_major_formatter(plt.ScalarFormatter())
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    pdf_filename = "plots/stochastic_ace_upsilon_sensitivity.png"
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    print(f"Plot saved successfully as: {pdf_filename}")

    plt.show()

def run_reproducible_campaign(tickers, chunk_size=4):
    T = 1500
    start_date = "2018-01-01"
    end_date = "2024-01-01"
    
    results = []
    ticker_items = list(tickers.items()) # Convert dictionary to a list of tuples for easy chunking
    
    print(f"Starting Campaign (T={T}, Dates: {start_date} to {end_date})...\n")
    
    # Loop over the tickers in chunks
    for chunk_idx in range(0, len(ticker_items), chunk_size):
        chunk = ticker_items[chunk_idx : chunk_idx + chunk_size]
        actual_chunk_size = len(chunk)
        part_number = (chunk_idx // chunk_size) + 1
        
        print(f"\n--- Generating Plot Part {part_number} ({actual_chunk_size} assets) ---")
        
        # Initialize a new figure just for this chunk
        plt.close('all')
        fig, axs = plt.subplots(2, actual_chunk_size, figsize=(5 * actual_chunk_size, 8))
        fig.suptitle(f'Real-World Application: Expanding Momentum Ensembles (Part {part_number})', fontsize=18, fontweight='bold')
        
        # If the chunk only has 1 item, axs is 1D. We need to standardize it to 2D for the loop.
        if actual_chunk_size == 1:
            axs = np.array([[axs[0]], [axs[1]]])
            
        for i, (ticker, name) in enumerate(chunk):
            print(f"Processing {name} ({ticker})...")
            env = ReproducibleMarketInstance(ticker, start_date, end_date, T)
            
            # --- Run LCB ---
            agent_lcb = LCBAgent(T)
            env.t = 0
            regret_lcb, chosen_lcb = evaluate_agent_stochastic(agent_lcb, env, T)
            switches_lcb_traj = np.concatenate(([0], np.cumsum(chosen_lcb[1:] != chosen_lcb[:-1])))
            total_switches_lcb = switches_lcb_traj[-1] 
            
            # --- Run UCB ---
            agent_ucb = UCBAgent(T)
            env.t = 0
            regret_ucb, chosen_ucb = evaluate_agent_stochastic(agent_ucb, env, T)
            switches_ucb_traj = np.concatenate(([0], np.cumsum(chosen_ucb[1:] != chosen_ucb[:-1])))
            total_switches_ucb = switches_ucb_traj[-1]
            
            # --- NEW: Extract True Optimal Trajectory ---
            opt_switches_traj = np.concatenate(([0], np.cumsum(env.optimal_expert_indices[1:] != env.optimal_expert_indices[:-1])))
            total_opt_switches = opt_switches_traj[-1]
            
            results.append({
                "Asset": name,
                "OPT Switches": total_opt_switches, # Add to results dictionary
                "LCB Regret": regret_lcb[-1], "LCB Switches": total_switches_lcb,
                "UCB Regret": regret_ucb[-1], "UCB Switches": total_switches_ucb
            })

            axs[0, i].plot(regret_lcb, label='LCB', color='blue', linewidth=2)
            axs[0, i].plot(regret_ucb, label='UCB', color='red', linewidth=2, linestyle='--')
            axs[0, i].set_title(f'{name}', fontsize=14)
            axs[0, i].set_xlabel('Trading Days (t)')
            axs[0, i].set_ylabel('Cumulative Regret' if i == 0 else '')
            axs[0, i].grid(True, alpha=0.3)
            if i == 0: 
                axs[0, i].legend(loc='upper left')

            axs[1, i].plot(switches_lcb_traj, label='LCB', color='blue', linewidth=2)
            axs[1, i].plot(switches_ucb_traj, label='UCB', color='red', linewidth=2, linestyle='--')
            axs[1, i].plot(opt_switches_traj, label='True OPT', color='green', linewidth=2, linestyle=':')
            axs[1, i].set_xlabel('Trading Days (t)')
            axs[1, i].set_ylabel('Number of Switches' if i == 0 else '')
            
            axs[1, i].set_yscale('symlog')
            axs[1, i].set_yticks([0, 10, 100, 1000, T])
            axs[1, i].get_yaxis().set_major_formatter(plt.ScalarFormatter())
            axs[1, i].grid(True, alpha=0.3)
            if i == 0:
                axs[1, i].legend(loc='upper left')
                    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf_filename = f"plots/real_world_financial_part_{part_number}.png"
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        print(f"Plot saved successfully as: {pdf_filename}")
        plt.show() 
    
    # --- Generate the Unified Markdown Table ---
    md = f"### Table X: Real-World Performance on Expanding Momentum Ensembles (T={T})\n\n"
    md += "| Asset | $\\pi_{LCB}$ Regret | $\\pi_{UCB}$ Regret | True OPT Switches ($\\Upsilon^*$) | $\\pi_{LCB}$ Switches | $\\pi_{UCB}$ Switches |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for r in results:
        md += f"| **{r['Asset']}** | {r['LCB Regret']:.2f} | {r['UCB Regret']:.2f} | {r['OPT Switches']} | **{r['LCB Switches']}** | {r['UCB Switches']} |\n"
        
    print("\n" + "="*50 + "\n")
    print(md)