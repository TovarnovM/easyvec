from easyvec import Vec2

def test_Vec2_constructor():
    v = Vec2(1,2)
    assert v is not None

def test_Vec2_cmp():
    from easyvec.vectors import get_CMP_TOL
    v1 = Vec2(1,2)
    v2 = Vec2(1 + get_CMP_TOL()/10, 2 + get_CMP_TOL()/10)
    assert v1 == v2

def test_Vec2_cmp2():
    from easyvec.vectors import get_CMP_TOL
    v1 = Vec2(1,2)
    v2 = Vec2(1 + get_CMP_TOL()*2, 2 + get_CMP_TOL()/10)
    assert (v1 == v2) == False

def test_Vec2_cmp3():
    from easyvec.vectors import get_CMP_TOL
    v1 = Vec2(1,2)
    v2 = Vec2(1 + get_CMP_TOL()*2, 2 + get_CMP_TOL()/10)
    assert v1 != v2