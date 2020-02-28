from pytest import approx
from easyvec import Vec2, intersect, closest
import numpy as np
from easyvec.geometry import _sortreduce, Rect





def test_closest_p1():
    assert closest((2,0), (0,2), (0,0)) == (1,1)

def test_closest_p2():
    assert closest((2,0), (0,2), p=(0,0)) == (1,1)

def test_closest_p3():
    assert closest(line=((2,0), (0,2)), p=(0,0)) == (1,1)

def test_closest_p4():
    assert closest(segment=((2,0), (0,2)), p=(0,0)) == (1,1)

def test_closest_p5():
    assert closest(ray=((2,0), (0,2)), p=(0,0)) == (1,1)

def test_closest_p6():
    assert closest(ray=((2,0), (0,2)), p=(3,0)) == (2,0)

def test_closest_p7():
    assert closest(segment=((2,0), (0,2)), p=(0,3)) == (0,2)

def test_Rect_union1():
    r1 = Rect(0,0, 3,3)
    r2 = Rect(-1,-3, 2,2)
    assert r1 + r2 == Rect(-1,-3,3,3)

def test_Rect_union2():
    r1 = Rect(0,0, 3,3)
    assert r1 + (-1,-3) == Rect(-1,-3,3,3)

def test_Rect_union3():
    r1 = Rect(0,0, 3,3)
    assert r1 + Vec2(-1,-3) == Rect(-1,-3,3,3)

def test_Rect_union4():
    r1 = Rect(0,0, 3,3)
    assert r1 + [-1,-3] == Rect(-1,-3,3,3)

def test_Rect_intersect1():
    r1 = Rect(0,0, 3,3)
    r2 = Rect(-1,-3, 2,2)
    assert r1.intersect(r2) == Rect(0,0,2,2)

def test_Rect_intersect2():
    r1 = Rect(0,0, 3,3)
    r2 = Rect(-1,-3, -2,-3)
    assert r1.intersect(r2).is_null()

def test_Rect_intersect3():
    r1 = Rect(0,0, 3,3)
    r2 = Rect(-1,-3, 2,2)
    assert r1 * r2 == Rect(0,0,2,2)

def test_Rect_intersect4():
    r = Rect(1,1, 3,3)
    point = r.intersect((0,2), (2,2))
    assert point == (1,2)

def test_Rect_intersect5():
    r = Rect(1,1, 3,3)
    point = r.intersect((0,2), (10,2))
    assert point == (1,2)

def test_Rect_intersect6():
    r = Rect(1,1, 3,3)
    point = r.intersect((0,0), (4,4))
    assert point == (1,1)

def test_Rect_intersect7():
    r = Rect(1,1, 3,3)
    point = r.intersect((0,3), (4,3))
    assert point == (1,3)

def test_Rect_intersect8():
    r = Rect(1,1, 3,3)
    point = r.intersect((0,4), (4,0))
    assert point == (1,3)

def test_Rect_intersect9():
    r = Rect(1,1, 3,3)
    point = r.intersect(r=((0,4), (4,0)))
    assert point == (1,3)


def test_Rect_perimeter():
    for i in range(1000):
        args = np.random.uniform(-100, 100, 4)
        r = Rect(*args)
        assert r.perimeter() >= 0 

def test_Rect_area():
    for i in range(1000):
        args = np.random.uniform(-100, 100, 4)
        r = Rect(*args)
        assert r.area() >= 0 

def test_Rect_is_null():
    r = Rect.null()
    assert r.is_null()

def test_Rect_unpack1():
    for i in range(1000):
        args = np.random.uniform(-100, 100, 4)
        r = Rect(*args)
        x1, y1, x2, y2 = r
        assert min(args[[0,2]]) == approx(x1)
        assert min(args[[1,3]]) == approx(y1)
        assert max(args[[0,2]]) == approx(x2)
        assert max(args[[1,3]]) == approx(y2)

def test_Rect_unpack2():
    def foo(**kwargs):
        return kwargs
    for i in range(1000):
        args = np.random.uniform(-100, 100, 4)
        r = Rect(*args)
        d = foo(**r)
        assert min(args[[0,2]]) == approx(d['x1'])
        assert min(args[[1,3]]) == approx(d['y1'])
        assert max(args[[0,2]]) == approx(d['x2'])
        assert max(args[[1,3]]) == approx(d['y2'])


def test_Rect_as_tuple():
    for i in range(1000):
        args = np.random.uniform(-100, 100, 4)
        r = Rect(*args)
        assert (min(args[[0,2]]), min(args[[1,3]]), max(args[[0,2]]), max(args[[1,3]])) == approx(r.as_tuple())


def test_Rect_as_np():
    for i in range(1000):
        args = np.random.uniform(-100, 100, 4)
        r = Rect(*args)
        assert [min(args[[0,2]]), min(args[[1,3]]), max(args[[0,2]]), max(args[[1,3]])] == approx(r.as_np())

def test_Rect_copy():
    for i in range(1000):
        args = np.random.uniform(-100, 100, 4)
        r = Rect(*args)
        r2 = r.copy()
        assert r == r2

def test_Rect_init1():
    for i in range(1000):
        args = np.random.uniform(-100, 100, 4)
        r = Rect(*args)
        assert r is not None

def test_Rect_init2():
    for i in range(1000):
        args1 = np.random.uniform(-100, 100, 2)
        args2 = np.random.uniform(-100, 100, 2)
        r = Rect(args1, args2)
        assert r is not None

def test_Rect_init3():
    for i in range(1000):
        args1 = np.random.uniform(-100, 100, 4)
        r = Rect(args1)
        assert r is not None

def test_sortreduce1():
    for i in range(1000):
        n = np.random.randint(1,13)
        arr = np.random.uniform(-100, 100, n)
        inds = _sortreduce(arr.copy())
        assert len(np.unique(arr)) == len(inds)

def test_sortreduce2():
    for i in range(1000):
        n = np.random.randint(1,13)
        arr = np.random.randint(-3, 3, n)
        arr2 = np.array(arr, dtype=np.double)
        inds = _sortreduce(arr2)
        arr3 = arr[inds]
        assert len(np.unique(arr)) == len(inds)

def test_sortreduce3():
    for i in range(1000):
        n = np.random.randint(1,13)
        arr = np.random.uniform(-100, 100, n)
        inds = _sortreduce(arr.copy())
        inds_right = np.argsort(arr)
        assert inds_right == approx(inds)

def test_sortreduce4():
    for i in range(1000):
        n = np.random.randint(1,13)
        arr = np.random.randint(-3, 3, n)
        arr2 = np.array(arr, dtype=np.double)
        inds = _sortreduce(arr2)
        arr3 = arr[inds]
        assert len(np.unique(arr)) == len(np.unique(arr3))

def test_sortreduce5():
    arr = np.array([1.0, 1, 2, 2])
    inds = _sortreduce(arr.copy())
    assert arr[inds] == approx([1,2])

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