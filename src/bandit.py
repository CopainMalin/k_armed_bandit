from src.arms import Arm


class Bandit:
    __slots__ = {"__arms"}

    def __init__(self, arms: list[Arm]) -> None:
        self.__arms = arms

    def __repr__(self) -> str:
        return f"Bandit(num_arms = {len(self.__arms)})"

    def __len__(self) -> int:
        return len(self.__arms)

    @property
    def arms(self) -> list[Arm]:
        return self.__arms

    def get_reward(self, arm_number: int) -> float:
        return self.__arms[arm_number].generate_reward()

    def get_q_star(self, arm_number: int) -> float:
        return self.__arms[arm_number].q_star
