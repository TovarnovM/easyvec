from pytest import approx
from easyvec import Vec2, intersect
import numpy as np

def test_intersect1():
    def vp(x, y):
        return Vec2(x,y) if np.random.random() < 0.5 else (x,y)
    for i in range(7000):
        v = intersect(vp(0,0), vp(3,3), vp(2,0), vp(0,2))
        assert v == (1,1)

def test_intersect2():
    def vp(x, y):
        return Vec2(x, y) if np.random.random() < 0.5 else (x, y)
    variants = ['line', 'l', 'line1', 'l1', 'line2', 'l2', \
        'segment', 's', 'segment1', 's1', 'segment2', 's2', \
        'ray', 'r', 'ray1', 'r1', 'ray2', 'r2' ]
    for i in range(7000):
        kw = {np.random.choice(variants): (vp(2,0), vp(0,2))}
        v = intersect(vp(0,0), vp(3,3), **kw)
        assert v == (1,1)

def test_intersect3():
    def vp(x, y):
        return Vec2(x, y) if np.random.random() < 0.5 else (x, y)
    variants = ['line', 'l', 'line1', 'l1', 'line2', 'l2', \
        'segment', 's', 'segment1', 's1', 'segment2', 's2', \
        'ray', 'r', 'ray1', 'r1', 'ray2', 'r2' ]
    for i in range(7000):
        k1, k2 = np.random.choice(variants, size=2, replace=False)
        kw = {
            str(k1): (vp(2,0), vp(0,2)),
            str(k2): (vp(0,0), vp(3,3))}
        v = intersect(**kw)
        assert v == (1,1)

def test_intersect4():
    def vp(x, y):
        return Vec2(x, y) if np.random.random() < 0.5 else (x, y)
    variants = ['line', 'l', 'line1', 'l1', 'line2', 'l2', \
        'segment', 's', 'segment1', 's1', 'segment2', 's2', \
        'ray', 'r', 'ray1', 'r1', 'ray2', 'r2' ]
    for i in range(7000):
        k1, k2 = np.random.choice(variants, size=2, replace=False)
        kw = {
            str(k1): (vp(1,0), vp(4,3)),
            str(k2): (vp(0,0), vp(3,3))}
        v = intersect(**kw)
        assert v is None