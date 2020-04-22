from easyvec import Mat3, Vec3
import numpy as np
from pytest import approx

def test_mat3_is_eq():
    m = Mat3.eye()
    assert m == np.array([[1,0,0], [0,1,0], [0,0,1]])

def test_mat3_is_eq2():
    m = Mat3.eye()
    assert m != np.array([[1,0,1], [0,1,0], [0,0,1]])

def test_mat_mul1():
    for i in range(1000):
        args = np.random.uniform(-100,100,9)
        m = Mat3(*args)
        assert m * m._1 == Mat3.eye()