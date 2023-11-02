from src.arms import GaussianArm


def test_GaussianArm_qstar():
    arm = GaussianArm(loc=0, scale=1)
    assert arm.q_star == 0
