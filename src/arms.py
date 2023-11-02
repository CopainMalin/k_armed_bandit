from typing import Protocol


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
