from numpy import zeros, arange
from numpy.random import normal
import matplotlib.pyplot as plt
import seaborn as sns
import mplcyberpunk
from src.agents import EpsilonGreedyAgent, UCBAgent
from src.bandit import Bandit
from src.arms import GaussianArm

def benchmark_agents():
    realist_gains = zeros(1000)
    optimist_gains = zeros(1000)
    greedy_gains = zeros(1000)
    ucb_gains = zeros(1000)

    for i in range(100):
        bandit = Bandit(
        [
            GaussianArm(0, 1),
            *[
                GaussianArm(loc=random_number, scale=1)
                for random_number in range(10)
            ],
        ]
        )

        realist_agent = EpsilonGreedyAgent(n_arms=len(bandit), optimist=False, epsilon=.1)
        for _ in range(1000):
            realist_agent.pick_and_update(bandit)
        realist_gains += realist_agent.reward_evolution
            
        optimist_agent = EpsilonGreedyAgent(n_arms=len(bandit), optimist=True, epsilon=.1)
        for _ in range(1000):
            optimist_agent.pick_and_update(bandit)
        optimist_gains += optimist_agent.reward_evolution
        
        greedy_agent = EpsilonGreedyAgent(n_arms=len(bandit), optimist=True, epsilon=0)
        for _ in range(1000):
            greedy_agent.pick_and_update(bandit)
        greedy_gains += greedy_agent.reward_evolution
        
        ucb_agent = UCBAgent(n_arms=len(bandit), c=2)
        for t in range(1, 1001):
            ucb_agent.pick_and_update(bandit, timestep=t)
        ucb_gains += ucb_agent.reward_evolution
        
    realist_gains /= 100
    optimist_gains /= 100
    greedy_gains /= 100
    ucb_gains /= 100
    
    plt.style.use("cyberpunk")
    plt.figure(figsize = (15, 10))
    plt.title("Agents performances averaged on 100 iterations of the ten arms testbed", fontweight = "bold", color = "white", fontsize = 15)
    plt.plot(realist_gains, label = r"Realist Agent | $\epsilon = 0.1$")
    plt.plot(optimist_gains, label = r"Optimist Agent | $\epsilon = 0.1$")
    plt.plot(greedy_gains, label = r"Greedy Agent | $\epsilon = 0$")
    plt.plot(ucb_gains, label = r"UCB Agent | c = 2")
    plt.xlabel("Steps", fontweight = "bold", color = "white", fontsize = 13)
    plt.ylabel("Average reward", fontweight = "bold", color = "white", fontsize = 13)
    plt.legend(fontsize = 13)
    plt.savefig("plots/10_armed_testbed_perfs.png", bbox_inches = "tight")
    
    
def ten_arms_testbed_example():
    bandit = Bandit([GaussianArm(loc = random_number, scale = 1) for random_number in normal(loc = 0, scale = 1, size = 10)])

    num_samples = 200

    samples = [normal(arm.q_star, 1, num_samples) for arm in bandit.arms]

    labels = [f"Arm {i+1}" for i in range(len(bandit))]
    plt.style.use("cyberpunk")
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=samples, inner="quartile", alpha = 1)
    plt.xticks(arange(len(bandit)), labels)
    plt.xlabel('Arms')
    plt.ylabel('Value')
    plt.title('Distribution of each Arm in the 10-Armed Testbed', fontweight = "bold", color = "white", fontsize = 15)
    plt.savefig("plots/ten_arms_testbed.png", bbox_inches = "tight")
