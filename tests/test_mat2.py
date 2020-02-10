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
    m = Mat2()
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

