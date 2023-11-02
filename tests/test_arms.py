from src.arms import GaussianArm, UniformArm, BernouilliArm


class TestGaussianArm:
    def setup_method(self):
        self.arm = GaussianArm(loc=0, scale=1)

    def test_qstar(self):
        assert self.arm.q_star == 0

    def test_generate_reward(self):
        # Reward must be in 3sigma interval
        assert -3 <= self.arm.generate_reward() <= 3


class TestUniformArm:
    def setup_method(self):
        self.arm = UniformArm(lower=0, upper=1)

    def test_qstar(self):
        assert self.arm.q_star == 0.5

    def test_generate_reward(self):
        # Reward must be in the two bounds
        assert 0 <= self.arm.generate_reward() <= 1


class TestBernouilliArm:
    def setup_method(self):
        self.arm = BernouilliArm(p=0.5)

    def test_qstar(self):
        assert self.arm.q_star == 0.5

    def test_p(self):
        assert 0 <= self.arm.p <= 1

    def test_generate_reward(self):
        rw = self.arm.generate_reward()
        assert (rw == 0) | (rw == 1)
