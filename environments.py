import numpy as np
from abc import ABC, abstractmethod
import yfinance as yf
import pandas as pd

class ACE_Instance(ABC):
    def __init__(self, T):
        self.T = T
        self.t = 0
        self.true_means = np.zeros(T)
        self._build_instance()

        self.optimal_means = np.maximum.accumulate(self.true_means)

    @abstractmethod
    def _build_instance(self):
        pass

    def get_optimal_reward(self):
        """Returns the expected reward of the best available expert at the current round."""
        # Experts 0 to t-1 are awake at round t.
        if self.t == 0:
            return 0.0
        return self.optimal_means[self.t - 1]

    def step(self):
        """Advances the environment and returns rewards for ALL awake experts."""
        self.t += 1
        awake_means = self.true_means[:self.t]
        # Generate Bernoulli rewards
        rewards = np.random.binomial(1, awake_means)
        return rewards

class OptimismTrapInstance(ACE_Instance):
    def _build_instance(self):
        # The very first expert is the best
        self.true_means[0] = 0.9
        
        # All subsequent experts are slightly worse, acting as "distractors"
        self.true_means[1:] = 0.8

class FrequentSwitcherInstance(ACE_Instance):
    def __init__(self, T, upsilon_star=None, mean_lower=0.25, mean_upper=0.75):
        if upsilon_star:
            self.upsilon_star = int(upsilon_star)
        else:
            self.upsilon_star = int(np.sqrt(T))
        self.mean_lower = mean_lower
        self.mean_upper = mean_upper
        super().__init__(T) # This automatically calls _build_instance
        
    def _build_instance(self):
        # 1. Set every single expert to 0.0 initially
        self.true_means.fill(0.0) 
        
        # 2. Calculate the exact interval between switches
        switch_interval = max(1, self.T // self.upsilon_star)
        
        # 3. Generate exactly upsilon_star values linearly spaced from lower to upper bound
        optimal_values = np.linspace(self.mean_lower, self.mean_upper, self.upsilon_star)
        
        # 4. Plant only the optimal experts at their specific awakening rounds
        for j in range(self.upsilon_star):
            optimal_idx = j * switch_interval
            if optimal_idx < self.T:
                self.true_means[optimal_idx] = optimal_values[j]

class UniformRandomInstance(ACE_Instance):
    def __init__(self, T):
        super().__init__(T) # Calls _build_instance
        
    def _build_instance(self):
        self.true_means = np.random.uniform(0,1, size=self.T)

class FinancialMarketInstance: # Removed inheritance for a standalone data wrapper
    def __init__(self, ticker="^GSPC", T=1000):
        self.T = T
        self.t = 0
        self.ticker = ticker
        
        # 1. Fetch Real-World Data
        print(f"Downloading historical data for {ticker}...")
        # Download extra history to ensure we have enough data to calculate momentum
        stock_data = yf.download(ticker, period="10y", interval="1d", progress=False)
        closes = stock_data['Close'].values.flatten()
        
        # We need the most recent T+1 days
        recent_closes = closes[-(self.T + 1):]
        
        # Actual market directions (1 if went up, 0 if went down)
        # Target for day t is whether price at t+1 is higher than price at t
        self.actual_directions = (recent_closes[1:] > recent_closes[:-1]).astype(int)
        
        # 2. Build the T x T Reward Matrix
        print("Building the expanding expert reward matrix...")
        self.reward_matrix = np.zeros((self.T, self.T))
        
        for t in range(self.T):
            current_price = recent_closes[t]
            
            # For each awake expert i (where i goes from 0 to t)
            # Expert i has a lookback window of (i + 1) days
            for i in range(t + 1):
                lookback_price = closes[-(self.T + 1) + t - (i + 1)]
                
                # Expert's Prediction: Up (1) if current price > lookback price, else Down (0)
                expert_prediction = 1 if current_price > lookback_price else 0
                
                # Reward is 1 if the expert was right, 0 if wrong
                reward = 1 if expert_prediction == self.actual_directions[t] else 0
                self.reward_matrix[i, t] = reward
                
        # Precompute the optimal means (for regret calculation)
        # In a real environment, the true expected mean is unknown, but we can 
        # approximate it for the sake of plotting using a rolling accuracy 
        # or simply evaluate empirical regret against the best-in-hindsight expert.
        self.true_means = np.mean(self.reward_matrix, axis=1)
        self.optimal_means = np.maximum.accumulate(self.true_means)

    def get_optimal_reward(self):
        if self.t == 0:
            return 0.0
        return self.optimal_means[self.t - 1]

    def step(self):
        self.t += 1
        # Return the exact column of rewards for time t, for all currently awake experts
        awake_rewards = self.reward_matrix[:self.t, self.t - 1]
        return awake_rewards
    
class ReproducibleMarketInstance:
    def __init__(self, ticker, start_date, end_date, T):
        self.T = T
        self.t = 0
        self.ticker = ticker
        
        # 1. Fetch fixed, reproducible historical data
        # We fetch a bit of extra history before 'start_date' so the first few experts 
        # have enough lookback data to awaken properly.
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=T*2)
        stock_data = yf.download(ticker, start=extended_start.strftime('%Y-%m-%d'), 
                                 end=end_date, progress=False)
        
        closes = stock_data['Close'].values.flatten()
        
        # Extract the exact T days we will simulate over
        recent_closes = closes[-(self.T + 1):]
        self.actual_directions = (recent_closes[1:] > recent_closes[:-1]).astype(int)
        
        # 2. Build the expanding expert reward matrix
        self.reward_matrix = np.zeros((self.T, self.T))
        
        for t in range(self.T):
            current_price = recent_closes[t]
            for i in range(t + 1):
                lookback_price = closes[-(self.T + 1) + t - (i + 1)]
                expert_prediction = 1 if current_price > lookback_price else 0
                self.reward_matrix[i, t] = 1 if expert_prediction == self.actual_directions[t] else 0

        # 3. Precompute means for regret calculation     
        self.true_means = np.mean(self.reward_matrix, axis=1)
        self.optimal_means = np.maximum.accumulate(self.true_means)
        
        self.optimal_expert_indices = np.zeros(self.T, dtype=int)
        current_best_idx = 0
        for t in range(self.T):
            # If the newly awakened expert has a higher true mean, it becomes the new optimal
            if self.true_means[t] > self.true_means[current_best_idx]:
                current_best_idx = t
            self.optimal_expert_indices[t] = current_best_idx

    def get_optimal_reward(self):
        if self.t == 0:
            return 0.0
        return self.optimal_means[self.t - 1]

    def step(self):
        self.t += 1
        return self.reward_matrix[:self.t, self.t - 1]