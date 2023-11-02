from abc import ABC, abstractmethod
from numpy.random import uniform, randint
from numpy import argmax, zeros, ones, ndarray
from src.bandit import Bandit


class Agent(ABC):
    @abstractmethod
    def __init__(self, n_arms: int):
        pass

    @abstractmethod
    def _update_beliefs(self):
        pass

    @abstractmethod
    def pick_one_arm(self):
        pass

    @abstractmethod
    def pick_and_update(self):
        pass


class EpsilonGreedyAgent(Agent):
    __slots__ = {
        "epsilon",
        "_expected_q_stars",
        "n_arms",
        "_arms_taken",
        "_rewards_per_arms",
        "_rewards_over_time",
    }

    def __init__(self, n_arms: int, epsilon: float = 0.1, optimist: bool = False):
        if optimist:
            self._expected_q_stars = ones(n_arms) * 5
        else:
            self._expected_q_stars = zeros(n_arms)

        self.epsilon = epsilon
        self.n_arms = n_arms
        self._arms_taken = zeros(n_arms)
        self._rewards_per_arms = zeros(n_arms)
        self._rewards_over_time = []

    @property
    def expected_q_stars(self) -> ndarray:
        return self._expected_q_stars

    @property
    def arms_taken(self) -> ndarray:
        return self._arms_taken

    @property
    def rewards_per_arms(self) -> ndarray:
        return self._rewards_per_arms

    @property
    def reward_evolution(self) -> list:
        return self._rewards_over_time

    def pick_one_arm(self) -> int:
        if uniform(0, 1, size=None) <= self.epsilon:
            choice = randint(low=0, high=self.n_arms, size=None)
        else:
            choice = argmax(self._expected_q_stars)

        return int(choice)

    def _update_beliefs(self, reward: float, arm: int):
        self._arms_taken[arm] += 1
        self._rewards_per_arms[arm] += reward
        self._rewards_over_time.append(reward)
        self._expected_q_stars[arm] += (
            self._rewards_per_arms[arm] / self._arms_taken[arm]
        )

    def pick_and_update(self, bandit: Bandit):
        arm = self.pick_one_arm()
        rw = bandit.get_reward(arm_number=arm)
        self._update_beliefs(reward=rw, arm=arm)
