from src.agents import EpsilonGreedyAgent
from src.arms import GaussianArm
from src.bandit import Bandit
from src.agents import EpsilonGreedyAgent
from numpy import sum as nsum


class TestEpsilonGreedyAgent:
    def setup_method(self):
        self.bandit = Bandit(
            [
                GaussianArm(0, 1),
                *[
                    GaussianArm(loc=random_number, scale=1)
                    for random_number in range(10)
                ],
            ]
        )
        self.fool_agent = EpsilonGreedyAgent(
            n_arms=len(self.bandit), optimist=False, epsilon=1
        )
        self.optimist_agent = EpsilonGreedyAgent(n_arms=len(self.bandit), optimist=True)
        self.realist_agent = EpsilonGreedyAgent(n_arms=len(self.bandit), optimist=False)

    def test_realist_setup(self):
        assert nsum(self.realist_agent.expected_q_stars) == 0

    def test_optimism_setup(self):
        assert (
            nsum(self.optimist_agent.expected_q_stars)
            == 10 * self.optimist_agent.n_arms
        )

    def test_reward_optimist_gte_realist(self):
        for _ in range(1000):
            self.optimist_agent.pick_and_update(self.bandit)
            self.realist_agent.pick_and_update(self.bandit)
            self.fool_agent.pick_and_update(self.bandit)

        assert nsum(self.optimist_agent.reward_evolution) >= nsum(
            self.realist_agent.reward_evolution
        )

    def test_arms_taken_positive(self):
        assert all(taken_time >= 0 for taken_time in self.optimist_agent.arms_taken)
        assert all(taken_time >= 0 for taken_time in self.realist_agent.arms_taken)

    def test_randomness_of_the_choice(self):
        assert max(self.fool_agent.arms_taken) < 500
