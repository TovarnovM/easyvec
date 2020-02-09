from easyvec import Vec2
import numpy as np
from pytest import approx

def test_constructor():
    v = Vec2(1,2)
    assert v is not None

def test_cmp1():
    from easyvec.vectors import get_CMP_TOL
    v1 = Vec2(1,2)
    v2 = Vec2(1 + get_CMP_TOL()/10, 2 + get_CMP_TOL()/10)
    assert v1 == v2

def test_cmp2():
    from easyvec.vectors import get_CMP_TOL
    v1 = Vec2(1,2)
    v2 = Vec2(1 + get_CMP_TOL()*2, 2 + get_CMP_TOL()/10)
    assert (v1 == v2) == False

def test_cmp3():
    from easyvec.vectors import get_CMP_TOL
    v1 = Vec2(1,2)
    v2 = Vec2(1 + get_CMP_TOL()*2, 2 + get_CMP_TOL()/10)
    assert v1 != v2

def test_cmp4():
    assert Vec2(-1, 1.11) == (-1, 1.11)

def test_cmp5():
    assert Vec2(-1, 1.11) == [-1, 1.11]

def test_cmp6():
    assert Vec2(-1, 1.11) != np.array([-1, 1.111])

def test_add1():
    assert Vec2(2,3) + Vec2(3,2) == Vec2(5,5)

def test_add2():
    assert Vec2(2,3) + (3,2) == Vec2(5,5)

def test_add3():
    assert Vec2(2,3) + [3,2] == Vec2(5,5)

def test_add4():
    import numpy as np
    assert Vec2(2,3) + np.array([3,2]) == Vec2(5,5)

def test_add5():
    assert Vec2(2,3) + 3 == Vec2(5,6)

def test_add6():
    import numpy as np
    assert [33,22] + Vec2(2,3) == Vec2(35,25)

def test_add7():
    assert (3,2) + Vec2(2,3)  == Vec2(5,5)

def test_add8():
    import numpy as np
    assert np.array([3,2]) + Vec2(2,3).as_np() == Vec2(5,5)

def test_add9():
    assert 3 + Vec2(2,3) == Vec2(5,6)

def test_add10():
    assert [33,22] + Vec2(2,3) == Vec2(35,25)

def test_unpack1():
    x, y = Vec2(-7,8)
    assert x == -7
    assert y == 8

def test_unpack2():
    def foo(x, y):
        return x, y
    x,y = foo(*Vec2(-7,8))
    assert x == -7
    assert y == 8

def test_unpack4():
    def foo(x, y):
        return x, y
    x,y = foo(**Vec2(-7,8))
    assert x == -7
    assert y == 8

def test_sub1():
    assert Vec2(1,2) - Vec2(3,5) == (-2, -3)

def test_sub2():
    assert Vec2(1, 2) - (3, 5) == [-2, -3]

def test_sub3():
    assert (1, 2) - Vec2(3, 5) == (-2, -3)

def test_sub4():
    assert Vec2(3, 5) - [1, 6] == (2, -1)

def test_sub5():
    assert [1, 6] - Vec2(3, 5) == (-2, 1)

def test_sub6():
    import numpy as np
    assert Vec2(1, 2) - np.array([3, 5]) == (-2, -3)

def test_sub7():
    assert 6 - Vec2(3, 5) == (3, 1)

def test_sub8():
    assert Vec2(3, 5) - 7 == (-4, -2)

def test_iadd1():
    v = Vec2(1, 2)
    v += Vec2(3, 5)
    assert v == Vec2(4, 7)

def test_iadd2():
    v = Vec2(1, 2)
    v += (3, 5)
    assert v == (4, 7)

def test_iadd3():
    v = Vec2(1, 2)
    v += 3
    assert v == (4, 5)

def test_iadd4():
    v = Vec2(1, 2)
    v += [3, 10]
    assert v == (4, 12)

def test_iadd5():
    import numpy as np
    v = Vec2(1, 2)
    v += np.array([3, 10])
    assert v == (4, 12)

def test_isub1():
    v = Vec2(1, 2)
    v -= Vec2(3, 5)
    assert v == Vec2(-2, -3)

def test_isub2():
    v = Vec2(1, 2)
    v -= (3, 5)
    assert v == (-2, -3)

def test_isub3():
    v = Vec2(1, 2)
    v -= 3
    assert v == (-2, -1)

def test_isub4():
    v = Vec2(1, 2)
    v -= [3, 10]
    assert v == [-2, -8]

def test_isub5():
    import numpy as np
    v = Vec2(1, 2)
    v -= np.array([3, 10])
    assert v == (-2, -8)

def test_mul1():
    assert Vec2(1, 2) * 3 == (3,6)

def test_mul2():
    assert Vec2(1, 2) * (2, -1) == approx(0)

def test_mul3():
    assert Vec2(1, 2) * [2, -1] == approx(0)

def test_mul4():
    assert Vec2(1, 2) * Vec2(2, -1) == approx(0)   

def test_mul5():
    import numpy as np
    assert Vec2(1, 2) * np.array([2, -1]) == approx(0)   

def test_mul11():
    assert 3 * Vec2(1, 2) == (3,6)

def test_mul21():
    assert (2, -1) * Vec2(1, 2)  == approx(0)

def test_mul31():
    assert [2, -1] * Vec2(1, 2) == approx(0)

def test_imul1():
    v = Vec2(1, 2)
    v *= Vec2(3, 5)
    assert v == Vec2(3, 10)

def test_imul2():
    v = Vec2(1, 2)
    v *= (3, 5)
    assert v == (3, 10)

def test_imul3():
    v = Vec2(1, 2)
    v *= 3
    assert v == (3, 6)

def test_imul4():
    v = Vec2(1, 2)
    v *= [3, 10]
    assert v == [3, 20]

def test_imul5():
    import numpy as np
    v = Vec2(1, 2)
    v *= np.array([3, 10])
    assert v == (3, 20)

def test_div1():
    assert Vec2(3, 6) / 3 == (1,2)

def test_div2():
    assert Vec2(10, 2) / (2, -1) == (5,-2)

def test_div3():
    assert Vec2(10, 2) / [2, -1] == (5,-2)

def test_div4():
    assert Vec2(10, 2) / Vec2(2, -1) == (5,-2)  

def test_div5():
    import numpy as np
    assert Vec2(10, 2) / np.array([2, -1]) == (5,-2)

def test_div11():
    assert 12 / Vec2(3, 4) == (4,3)

def test_div21():
    assert (12, 8) / Vec2(3, 4) == (4,2)

def test_div31():
    assert [12, 8] / Vec2(3, 4) == (4,2)

def test_idiv1():
    v = Vec2(12, 10)
    v /= Vec2(3, 5)
    assert v == Vec2(4, 2)

def test_idiv2():
    v = Vec2(12, 10)
    v /= (3, 5)
    assert v == (4, 2)

def test_idiv3():
    v = Vec2(12, 15)
    v /= 3
    assert v == (4, 5)

def test_idiv4():
    v = Vec2(12, 20)
    v /= [3, 10]
    assert v == [4, 2]

def test_idiv5():
    import numpy as np
    v = Vec2(12, 20)
    v /= np.array([3, 10])
    assert v == (4, 2)

def test_floordiv1():
    assert Vec2(4, 8) // 3 == (1,2)

def test_floordiv2():
    assert Vec2(11, 2.5) // (2, -1) == (5,-3)

def test_floordiv3():
    assert Vec2(11, 2.5) // [2, -1] == (5,-3)

def test_floordiv4():
    assert Vec2(11, 2.5) // Vec2(2, -1) == (5,-3)  

def test_floordiv5():
    import numpy as np
    assert Vec2(11, 2.5) // np.array([2, -1]) == (5,-3)

def test_floordiv11():
    assert 12 // Vec2(3, 4) == (4,3)

def test_floordiv21():
    assert (13, 9) // Vec2(3, 4) == (4,2)

def test_floordiv31():
    assert [13, 9] // Vec2(3, 4) == (4,2)

def test_ifloordiv1():
    v = Vec2(13, 12)
    v //= Vec2(3, 5)
    assert v == Vec2(4, 2)

def test_ifloordiv2():
    v = Vec2(13, 12)
    v //= (3, 5)
    assert v == (4, 2)

def test_ifloordiv3():
    v = Vec2(13, 17)
    v //= 3
    assert v == (4, 5)

def test_ifloordiv4():
    v = Vec2(13, 24)
    v //= [3, 10]
    assert v == [4, 2]

def test_ifloordiv5():
    import numpy as np
    v = Vec2(13, 24)
    v //= np.array([3, 10])
    assert v == (4, 2)

def test_mod1():
    assert Vec2(4, 8) % 3 == (1,2)

def test_mod2():
    assert Vec2(11, 2.1) % (2, -1) == (1,0.1)

def test_mod3():
    assert Vec2(11, 2.9) % [2, -1] == (1,0.9)

def test_mod4():
    assert Vec2(11, 2.1) % Vec2(2, -1) == (1,0.1)  

def test_mod5():
    import numpy as np
    assert Vec2(11, 2.1) % np.array([2, -1]) == (1,0.1)

def test_mod11():
    assert 13 % Vec2(3, 5) == (1,3)

def test_mod21():
    assert (13, 10) % Vec2(3, 4) == (1,2)

def test_mod31():
    assert [13, 9] % Vec2(3, 4) == (1,1)

def test_imod1():
    v = Vec2(13, 12)
    v %= Vec2(3, 5)
    assert v == Vec2(1, 2)

def test_imod2():
    v = Vec2(13, 12)
    v %= (3, 5)
    assert v == (1, 2)

def test_imod3():
    v = Vec2(13, 17)
    v %= 3
    assert v == (1, 2)

def test_imod4():
    v = Vec2(13, 24)
    v %= [3, 10]
    assert v == [1, 4]

def test_imod5():
    import numpy as np
    v = Vec2(13, 24)
    v %= np.array([3, 10])
    assert v == (1, 4)