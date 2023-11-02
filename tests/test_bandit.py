from src.arms import GaussianArm
from numpy.random import normal
from src.bandit import Bandit
import pytest


@pytest.fixture
def generate_bandit() -> Bandit:
    return Bandit(
        [
            GaussianArm(0, 1),
            *[
                GaussianArm(loc=random_number, scale=1)
                for random_number in normal(loc=0, scale=5, size=9)
            ],
        ]
    )


def test_get_reward(generate_bandit):
    assert -3 <= generate_bandit.get_reward(0) <= 3


def test_len(generate_bandit):
    assert len(generate_bandit) == 10


def test_get_qstar(generate_bandit):
    assert generate_bandit.get_q_star(0) == 0
