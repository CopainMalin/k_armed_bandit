from typing import Protocol
from numpy.random import normal, uniform, binomial


class Arm(Protocol):
    def __init__(self) -> None:
        ...

    @property
    def q_star() -> float:
        ...

    def generate_reward() -> float:
        ...


class GaussianArm:
    __slots__ = {"loc", "_scale"}

    def __init__(self, loc: float, scale: float) -> None:
        self.loc = loc
        self._scale = scale

    @property
    def q_star(self) -> float:
        return self.loc

    def generate_reward(self) -> float:
        return normal(loc=self.loc, scale=self._scale, size=None)

    def __repr__(self) -> str:
        return f"N({self.loc:.2f}, {self._scale:.2f})"


class UniformArm:
    __slots__ = {"upper", "lower"}

    def __init__(self, upper: float, lower: float) -> None:
        self.upper = upper
        self.lower = lower

    def __repr__(self) -> str:
        return f"U({self.lower:.2f}, {self.upper:.2f})"

    @property
    def q_star(self) -> float:
        return (self.upper + self.lower) / 2

    def generate_reward(self) -> float:
        return uniform(low=self.lower, high=self.upper, size=None)


class BernouilliArm:
    __slots__ = {"p"}

    def __init__(self, p: float) -> None:
        self.p = p

    @property
    def q_star(self) -> float:
        return self.p

    def generate_reward(self) -> float:
        return binomial(n=1, p=self.p, size=None)

    def __repr__(self) -> str:
        return f"B({self.p:.2f})"
