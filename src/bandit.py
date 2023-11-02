from src.arms import Arm


class Bandit:
    __slots__ = {"__arms"}

    def __init__(self, arms: list[Arm]) -> None:
        self.__arms = arms

    def __repr__(self) -> list[str]:
        return "[" + ", ".join(map(str, self.__arms)) + "]"

    def get_reward(self, arm_number: int) -> float:
        return self.__arms[arm_number].generate_reward()

    def get_q_star(self, arm_number: int) -> float:
        return self.__arms[arm_number].q_star
