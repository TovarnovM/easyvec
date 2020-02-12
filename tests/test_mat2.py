from easyvec import Mat2, Vec2
import numpy as np
from pytest import approx

def test_constructor1():
    m = Mat2(1,2,3,4)
    assert m is not None
    assert m.m11 == approx(1)
    assert m.m12 == approx(2)
    assert m.m21 == approx(3)
    assert m.m22 == approx(4) 

def test_constructor2():
    m = Mat2([1,2,3,4])
    assert m is not None
    assert m.m11 == approx(1)
    assert m.m12 == approx(2)
    assert m.m21 == approx(3)
    assert m.m22 == approx(4) 

def test_constructor3():
    m = Mat2([[1,2],[3,4]])
    assert m is not None
    assert m.m11 == approx(1)
    assert m.m12 == approx(2)
    assert m.m21 == approx(3)
    assert m.m22 == approx(4) 

def test_constructor4():
    m = Mat2([1,2],[3,4])
    assert m is not None
    assert m.m11 == approx(1)
    assert m.m12 == approx(2)
    assert m.m21 == approx(3)
    assert m.m22 == approx(4) 

def test_constructor5():
    m = Mat2(Vec2(1,2),Vec2(3,4))
    assert m is not None
    assert m.m11 == approx(1)
    assert m.m12 == approx(2)
    assert m.m21 == approx(3)
    assert m.m22 == approx(4) 

def test_constructor6():
    m = Mat2([Vec2(1,2),Vec2(3,4)])
    assert m is not None
    assert m.m11 == approx(1)
    assert m.m12 == approx(2)
    assert m.m21 == approx(3)
    assert m.m22 == approx(4) 


def test_constructor7():
    m = Mat2.eye()
    assert m is not None
    assert m.m11 == approx(1)
    assert m.m12 == approx(0)
    assert m.m21 == approx(0)
    assert m.m22 == approx(1) 

def test_constructor8():
    from math import sin, cos, pi
    for angle in np.random.uniform(-720, 720, 1000):
        angle *= pi/180
        m = Mat2.from_angle(angle)
        assert m is not None
        assert m.m11 == approx(cos(angle))
        assert m.m12 == approx(sin(angle))
        assert m.m21 == approx(-sin(angle))
        assert m.m22 == approx(cos(angle)) 

def test_constructor9():
    m = Mat2.from_xaxis((1,1))
    assert m is not None
    assert m.m11 == approx(1/2**0.5)
    assert m.m12 == approx(1/2**0.5)
    assert m.m21 == approx(-1/2**0.5)
    assert m.m22 == approx(1/2**0.5) 

def test_xiyj_axis():
    m = Mat2(1,2,3,4)
    assert m.x_axis() == (1,2)
    assert m.i_axis() == (1,2)
    assert m.y_axis() == (3,4)
    assert m.j_axis() == (3,4)

def test_cmp():
    m = Mat2(-1,2,-3,4)
    assert m == [[-1,2],[-3,4]]
    assert m != [[-1,-2],[-3,4]]

def test_T():
    m = Mat2(-1,2,-3,4)
    assert m.T == [[-1,-3], [2,4]]

def test_inverse1():
    for angle in np.random.uniform(-720,720,1000):
        m = Mat2.from_angle(angle)
        assert m._1 == m.T
        assert m.det() == approx(1)


def test_inverse2():
    for ms in np.random.uniform(-720,720,(1000,4)):
        m = Mat2(ms)
        if abs(m.det()) < 1e-6:
            continue 
        assert m * m._1 == Mat2.eye()

def test_mul1():
    for ms in np.random.uniform(-720,720,(1000,5)):
        m = Mat2(ms[:-1])
        assert m * ms[-1] == (ms[:-1] * ms[-1]).reshape(2,2)
        assert ms[-1] * m  == (ms[:-1] * ms[-1]).reshape(2,2)

def test_mul2():
    for angle, x, y in np.random.uniform(-180,180,(1000,3)):
        m = Mat2.from_angle(angle, 1)
        v = Vec2(x, y).norm()
        v1 = m * v
        assert v.angle_to(v1, 1) == approx(-angle)
        v2 = m._1 * v1
        assert v2 == v
        v3 = m._1 * v
        assert v.angle_to(v3, 1) == approx(angle) 

def test_imul():
    for ms in np.random.uniform(-720,720,(1000,4)):
        m = Mat2(ms)
        if abs(m.det()) < 1e-6:
            continue 
        assert m * m._1 == Mat2.eye()
        m *= m._1
        assert m == Mat2.eye()

def test_add1():
    for ms in np.random.uniform(-720,720,(1000,5)):
        m = Mat2(ms[:-1])
        m1 = m + ms[-1]
        m1i = (ms[:-1] + ms[-1]).reshape(2,2)
        assert m1 == m1i
        assert ms[-1] + m  == (ms[:-1] + ms[-1]).reshape(2,2)

def test_add2():
    for ms in np.random.uniform(-720,720,(1000,8)):
        m1 = Mat2(ms[:4])
        m2 = Mat2(ms[4:])
        assert m1 + m2 == m2 + m1
        assert m1 + m2 == (ms[:4] + ms[4:]).reshape(2,2)

def test_iadd():
    for ms in np.random.uniform(-720,720,(1000,8)):
        m1 = Mat2(ms[:4])
        m2 = Mat2(ms[4:])
        m12 = m1 + m2
        m1 += m2
        assert m12 == m1
        assert m1 == (ms[:4] + ms[4:]).reshape(2,2)

def test_sub1():
    for ms in np.random.uniform(-720,720,(1000,5)):
        m = Mat2(ms[:-1])
        m1 = m - ms[-1]
        m1i = (ms[:-1] - ms[-1]).reshape(2,2)
        assert m1 == m1i
        assert ms[-1] - m  == -(ms[:-1] - ms[-1]).reshape(2,2)

def test_sub2():
    for ms in np.random.uniform(-720,720,(1000,8)):
        m1 = Mat2(ms[:4])
        m2 = Mat2(ms[4:])
        assert m1 - m2 == -(m2 - m1)
        assert m1 - m2 == (ms[:4] - ms[4:]).reshape(2,2)

def test_isub():
    for ms in np.random.uniform(-720,720,(1000,8)):
        m1 = Mat2(ms[:4])
        m2 = Mat2(ms[4:])
        m12 = m1 - m2
        m1 -= m2
        assert m12 == m1
        assert m1 == (ms[:4] - ms[4:]).reshape(2,2)

def test_div1():
    for ms in np.random.uniform(-720,720,(1000,5)):
        m = Mat2(ms[:-1])
        m1 = m / ms[-1]
        m1i = (ms[:-1] / ms[-1]).reshape(2,2)
        assert m1 == m1i

def test_div2():
    for ms in np.random.uniform(-720,720,(1000,8)):
        m1 = Mat2(ms[:4])
        m2 = Mat2(ms[4:])
        assert m1 / m2 == (ms[:4] / ms[4:]).reshape(2,2)

def test_idiv():
    for ms in np.random.uniform(-720,720,(1000,8)):
        m1 = Mat2(ms[:4])
        m2 = Mat2(ms[4:])
        m12 = m1 / m2
        m1 /= m2
        assert m12 == m1
        assert m1 == (ms[:4] / ms[4:]).reshape(2,2)