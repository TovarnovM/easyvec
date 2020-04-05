
from libc.math cimport fabs, sqrt, round, ceil, floor, trunc, pi, atan2, sin, cos
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_GE, Py_GT, Py_NE
cimport cython
import numpy as np


CMP_TOL = 1e-8
BIG_REAL = 1e33
MINUS_BIG_REAL = -1e33

def get_CMP_TOL():
    return CMP_TOL

def get_BIG_REAL():
    return BIG_REAL

def get_MINUS_BIG_REAL():
    return MINUS_BIG_REAL

@cython.final
cdef class Vec2:
    @classmethod
    def from_list(cls, lst):
        return cls(lst[0], lst[1])

    @classmethod
    def from_dict(cls, dct):
        return cls(dct['x'], dct['y'])

    @classmethod
    def random(cls, p1, p2):
        x1 = min(p1[0], p2[0])
        x2 = max(p1[0], p2[0])
        y1 = min(p1[1], p2[1])
        y2 = max(p1[1], p2[1])
        x = np.random.uniform(x1,x2)
        y = np.random.uniform(y1,y2)
        return cls(x, y)

    def __cinit__(self, real x, real y):
        self.x = x
        self.y = y

    cpdef Vec2 clone(self):
        return Vec2(self.x, self.y)

    cpdef Vec2 copy(self):
        return Vec2(self.x, self.y)

    def to_dict(self) -> dict:
        return {k: self[k] for k in self.keys()}

    def __str__(self):
        return f'({self.x:.2f}, {self.y:.2f})'

    def __repr__(self):
        return f'Vec2({self.x}, {self.y})'

    @cython.nonecheck(False)
    def __richcmp__(v1, v2, int op):
        if op == Py_EQ:
            return fabs(v1[0] - v2[0]) < CMP_TOL and fabs(v1[1] - v2[1]) < CMP_TOL 
        elif op == Py_NE:
            return fabs(v1[0] - v2[0]) >= CMP_TOL or fabs(v1[1] - v2[1]) >= CMP_TOL
        raise NotImplementedError("Такой тип сравнения не поддерживается")

    cpdef bint is_eq(self, Vec2 other):
        return fabs(self.x - other.x) < CMP_TOL and fabs(self.y - other.y) < CMP_TOL

    cpdef Vec2 add_num_(self, real num):
        self.x += num
        self.y += num
        return self
    
    cpdef Vec2 add_num(self, real num):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.add_num_(num)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 add_vec_(self, Vec2 vec):
        self.x += vec.x
        self.y += vec.y
        return self

    @cython.nonecheck(False)
    cpdef Vec2 add(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.add_vec_(vec)
        return result   

    @cython.nonecheck(False)
    cpdef Vec2 add_(self, Vec2 vec):
        self.x += vec.x
        self.y += vec.y
        return self

    @cython.nonecheck(False)
    cpdef Vec2 add_vec(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.add_vec_(vec)
        return result

    cpdef Vec2 add_xy_(self, real x, real y):
        self.x += x
        self.y += y
        return self

    cpdef Vec2 add_xy(self, real x, real y):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.add_xy_(x, y)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 add_tup_(self, tuple tup):
        self.x += <real>(tup[0])
        self.y += <real>(tup[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 add_tup(self, tuple tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.add_tup_(tup)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 add_list_(self, list tup):
        self.x += <real>(tup[0])
        self.y += <real>(tup[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 add_list(self, list tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.add_list_(tup)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 add_arr_(self, rational[:] arr):
        self.x += <real>(arr[0])
        self.y += <real>(arr[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 add_arr(self, rational[:] tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.add_arr_(tup)
        return result

    def __add__(left, right) -> Vec2:
        if isinstance(left, Vec2):
            if isinstance(right, Vec2):
                return (<Vec2>left).add_vec(<Vec2>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview):
                return (<Vec2>left).add_xy(<real>(right[0]), <real>(right[1]))
            elif isinstance(right, int) or isinstance(right, float):
                return (<Vec2>left).add_num(<real>right)
        elif isinstance(right, Vec2):
            if isinstance(left, np.ndarray) or isinstance(left, tuple) or isinstance(left, list) or isinstance(left, memoryview):
                return (<Vec2>right).add_xy(<real>(left[0]), <real>(left[1]))
            elif isinstance(left, int) or isinstance(left, float):
                return (<Vec2>right).add_num(<real>left)         

        raise NotImplementedError(f"Складывать данные сущности нельзя left={left}, right={right}")

    cpdef Vec2 neg_(self):
        self.x = -self.x
        self.y = -self.y
        return self

    cpdef Vec2 neg(self):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.neg_()
        return result

    def __neg__(self) -> Vec2:
        cdef Vec2 result = Vec2(self.x, self.y)
        result.neg_()
        return result
    
    def __iadd__(self, other):
        if isinstance(other, Vec2):
            self.add_vec_(<Vec2>other)
        elif isinstance(other, np.ndarray) or isinstance(other, tuple) or isinstance(other, list) or isinstance(other, memoryview):
            self.add_xy_(<real>(other[0]), <real>(other[1]))
        elif isinstance(other, int) or isinstance(other, float):
            self.add_num_(<real>other)
        else:
            NotImplementedError(f"Прибавить данную сущность нельзя other={other}")
        return self

    def __getitem__(self, key) -> real:
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key =='x':
            return self.x
        elif key == 'y':
            return self.y
        raise IndexError(f'Невозможно получить компонент вектора по индексу {key}')

    def __setitem__(self, key, value: real):
        if key == 0:
            self.x = <real>value
        elif key == 1:
            self.y = <real>value
        elif key =='x':
            self.x = <real>value
        elif key == 'y':
            self.y = <real>value
        else:
            raise IndexError(f'Невозможно получить компонент вектора по индексу {key}')
    
    cpdef list keys(self):
        return ['x', 'y'] 

    def __iter__(self):
        return self.get_generator()

    def get_generator(self):
        yield self.x
        yield self.y

    def as_np(self):
        return np.array([self.x, self.y])
    
    cpdef tuple as_tuple(self):
        return (self.x, self.y)

    cpdef Vec2 sub_num_(self, real num):
        self.x -= num
        self.y -= num
        return self

    cpdef Vec2 sub_num(self, real num):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.sub_num_(num)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 sub_vec_(self, Vec2 vec):
        self.x -= vec.x
        self.y -= vec.y
        return self

    @cython.nonecheck(False)
    cpdef Vec2 sub_vec(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.sub_vec_(vec)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 sub_(self, Vec2 vec):
        self.x -= vec.x
        self.y -= vec.y
        return self

    @cython.nonecheck(False)
    cpdef Vec2 sub(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.sub_vec_(vec)
        return result

    cpdef Vec2 sub_xy_(self, real x, real y):
        self.x -= x
        self.y -= y
        return self

    cpdef Vec2 sub_xy(self, real x, real y):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.sub_xy_(x, y)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 sub_tup_(self, tuple tup):
        self.x -= <real>(tup[0])
        self.y -= <real>(tup[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 sub_tup(self, tuple tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.sub_tup_(tup)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 sub_list_(self, list tup):
        self.x -= <real>(tup[0])
        self.y -= <real>(tup[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 sub_list(self, list tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.sub_list_(tup)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 sub_arr_(self, rational[:] arr):
        self.x -= <real>(arr[0])
        self.y -= <real>(arr[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 sub_arr(self, rational[:] tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.sub_arr_(tup)
        return result

    def __sub__(left, right) -> Vec2:
        if isinstance(left, Vec2):
            if isinstance(right, Vec2):
                return (<Vec2>left).sub_vec(<Vec2>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview):
                return (<Vec2>left).sub_xy(<real>(right[0]), <real>(right[1]))
            elif isinstance(right, int) or isinstance(right, float):
                return (<Vec2>left).sub_num(<real>right)

        elif isinstance(right, Vec2):
            if isinstance(left, np.ndarray) or isinstance(left, tuple) or isinstance(left, list) or isinstance(left, memoryview):
                return Vec2(<real>(left[0]), <real>(left[1])).sub_vec_(<Vec2>right)  
            elif isinstance(left, int) or isinstance(left, float):
                return Vec2(<real>(left), <real>(left)).sub_vec_(<Vec2>right)        
        raise NotImplementedError(f"Вычитать данные сущности нельзя left={left}, right={right}")
    
    def __isub__(self, other) -> Vec2:
        if isinstance(other, Vec2):
            return self.sub_vec_(<Vec2>other)
        elif isinstance(other, np.ndarray) or isinstance(other, tuple) or isinstance(other, list) or isinstance(other, memoryview):
            return self.sub_xy_(<real>(other[0]), <real>(other[1]))
        elif isinstance(other, int) or isinstance(other, float):
            return self.sub_num_(<real>other)
        else:
            raise NotImplementedError(f"Вычитать данные сущности нельзя  other={other}")

    cpdef Vec2 mul_num_(self, real num):
        self.x *= num
        self.y *= num
        return self

    cpdef Vec2 mul_num(self, real num):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mul_num_(num)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 mul_vec_(self, Vec2 vec):
        self.x *= vec.x
        self.y *= vec.y
        return self

    @cython.nonecheck(False)
    cpdef Vec2 mul_vec(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mul_vec_(vec)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 mul_(self, Vec2 vec):
        self.x *= vec.x
        self.y *= vec.y
        return self

    @cython.nonecheck(False)
    cpdef Vec2 mul(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mul_vec_(vec)
        return result

    cpdef Vec2 mul_xy_(self, real x, real y):
        self.x *= x
        self.y *= y
        return self

    cpdef Vec2 mul_xy(self, real x, real y):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mul_xy_(x, y)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 mul_tup_(self, tuple vec):
        self.x *= <real>(vec[0])
        self.y *= <real>(vec[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 mul_tup(self, tuple vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mul_tup_(vec)
        return result   

    @cython.nonecheck(False)
    cpdef Vec2 mul_list_(self, list tup):
        self.x *= <real>(tup[0])
        self.y *= <real>(tup[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 mul_list(self, list tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mul_list_(tup)
        return result

    @cython.nonecheck(False)
    cpdef Vec2 mul_arr_(self, rational[:] arr):
        self.x *= <real>(arr[0])
        self.y *= <real>(arr[1])
        return self

    @cython.nonecheck(False)
    cpdef Vec2 mul_arr(self, rational[:] tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mul_arr_(tup)
        return result

    @cython.nonecheck(False)
    cpdef real dot(self, Vec2 vec):
        return self.x * vec.x + self.y * vec.y
    
    cpdef real dot_xy(self, real x, real y):
        return self.x * x + self.y * y

    @cython.nonecheck(False)
    cpdef real dot_tup(self, tuple tup):
        return self.x * <real>(tup[0]) + self.y * <real>(tup[1])

    @cython.nonecheck(False)
    cpdef real dot_list(self, list tup):
        return self.x * <real>(tup[0]) + self.y * <real>(tup[1])

    @cython.nonecheck(False)
    cpdef real dot_arr(self, rational[:] tup):
        return self.x * <real>(tup[0]) + self.y * <real>(tup[1])

    def __mul__(left, right):
        if isinstance(left, Vec2):
            if isinstance(right, Vec2):
                return (<Vec2>left).dot(<Vec2>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview) :
                return (<Vec2>left).dot_xy(<real>(right[0]), <real>(right[1]))
            elif isinstance(right, int) or isinstance(right, float):
                return (<Vec2>left).mul_num(<real>right)

        elif isinstance(right, Vec2):
            if isinstance(left, np.ndarray) or isinstance(left, tuple) or isinstance(left, list) or isinstance(left, memoryview):
                return (<Vec2>right).dot_xy(<real>(left[0]), <real>(left[1]))
            elif isinstance(left, int) or isinstance(left, float):
                return (<Vec2>right).mul_num(<real>left)     
        raise NotImplementedError(f"Перемножить данные сущности нельзя left={left}, right={right}")
    
    def __imul__(self, other) -> Vec2:
        if isinstance(other, Vec2):
            return self.mul_vec_(<Vec2>other)
        elif isinstance(other, np.ndarray) or isinstance(other, tuple) or isinstance(other, list) or isinstance(other, memoryview):
            return self.mul_xy_(<real>(other[0]), <real>(other[1]))
        elif isinstance(other, int) or isinstance(other, float):
            return self.mul_num_(<real>other)
        else:
            raise NotImplementedError(f"Домножить на данную сущность нельзя  other={other}")


    @cython.cdivision(True)
    cpdef Vec2 div_num_(self, real num):
        self.x /= num
        self.y /= num
        return self

    @cython.cdivision(True)
    cpdef Vec2 div_num(self, real num):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.div_num_(num)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_vec_(self, Vec2 vec):
        self.x /= vec.x
        self.y /= vec.y
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_vec(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.div_vec_(vec)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_(self, Vec2 vec):
        self.x /= vec.x
        self.y /= vec.y
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.div_vec_(vec)
        return result

    @cython.cdivision(True)
    cpdef Vec2 div_xy_(self, real x, real y):
        self.x /= x
        self.y /= y
        return self

    @cython.cdivision(True)
    cpdef Vec2 div_xy(self, real x, real y):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.div_xy_(x, y)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_tup_(self, tuple vec):
        self.x /= <real>(vec[0])
        self.y /= <real>(vec[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_tup(self, tuple vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.div_tup_(vec)
        return result   

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_list_(self, list tup):
        self.x /= <real>(tup[0])
        self.y /= <real>(tup[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_list(self, list tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.div_list_(tup)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_arr_(self, rational[:] arr):
        self.x /= <real>(arr[0])
        self.y /= <real>(arr[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 div_arr(self, rational[:] tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.div_arr_(tup)
        return result

    def __truediv__(left, right):
        if isinstance(left, Vec2):
            if isinstance(right, Vec2):
                return (<Vec2>left).div_vec(<Vec2>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview) :
                return (<Vec2>left).div_xy(<real>(right[0]), <real>(right[1]))
            elif isinstance(right, int) or isinstance(right, float):
                return (<Vec2>left).div_num(<real>right)

        elif isinstance(right, Vec2):
            if isinstance(left, np.ndarray) or isinstance(left, tuple) or isinstance(left, list) or isinstance(left, memoryview):
                return (Vec2(<real>(left[0]), <real>(left[1]))).div_vec_(<Vec2>right)
            elif isinstance(left, int) or isinstance(left, float):
                return Vec2(<real>left, <real>left).div_vec_(<Vec2>right)     
        raise NotImplementedError(f"Поделить данные сущности нельзя left={left}, right={right}")
    
    def __itruediv__(self, other) -> Vec2:
        if isinstance(other, Vec2):
            return self.div_vec_(<Vec2>other)
        elif isinstance(other, np.ndarray) or isinstance(other, tuple) or isinstance(other, list) or isinstance(other, memoryview):
            return self.div_xy_(<real>(other[0]), <real>(other[1]))
        elif isinstance(other, int) or isinstance(other, float):
            return self.div_num_(<real>other)
        else:
            raise NotImplementedError(f"Поделить на данную сущность нельзя  other={other}")


    @cython.cdivision(True)
    cpdef Vec2 floordiv_num_(self, real num):
        self.x //= num
        self.y //= num
        return self

    @cython.cdivision(True)
    cpdef Vec2 floordiv_num(self, real num):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.floordiv_num_(num)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_vec_(self, Vec2 vec):
        self.x //= vec.x
        self.y //= vec.y
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_vec(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.floordiv_vec_(vec)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_(self, Vec2 vec):
        self.x //= vec.x
        self.y //= vec.y
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.floordiv_vec_(vec)
        return result

    @cython.cdivision(True)
    cpdef Vec2 floordiv_xy_(self, real x, real y):
        self.x //= x
        self.y //= y
        return self

    @cython.cdivision(True)
    cpdef Vec2 floordiv_xy(self, real x, real y):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.floordiv_xy_(x, y)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_tup_(self, tuple vec):
        self.x //= <real>(vec[0])
        self.y //= <real>(vec[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_tup(self, tuple vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.floordiv_tup_(vec)
        return result   

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_list_(self, list tup):
        self.x //= <real>(tup[0])
        self.y //= <real>(tup[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_list(self, list tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.floordiv_list_(tup)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_arr_(self, rational[:] arr):
        self.x //= <real>(arr[0])
        self.y //= <real>(arr[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 floordiv_arr(self, rational[:] tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.floordiv_arr_(tup)
        return result

    def __floordiv__(left, right):
        if isinstance(left, Vec2):
            if isinstance(right, Vec2):
                return (<Vec2>left).floordiv_vec(<Vec2>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview) :
                return (<Vec2>left).floordiv_xy(<real>(right[0]), <real>(right[1]))
            elif isinstance(right, int) or isinstance(right, float):
                return (<Vec2>left).floordiv_num(<real>right)

        elif isinstance(right, Vec2):
            if isinstance(left, np.ndarray) or isinstance(left, tuple) or isinstance(left, list) or isinstance(left, memoryview):
                return (Vec2(<real>(left[0]), <real>(left[1]))).floordiv_vec_(<Vec2>right)
            elif isinstance(left, int) or isinstance(left, float):
                return Vec2(<real>left, <real>left).floordiv_vec_(<Vec2>right)     
        raise NotImplementedError(f"Поделить данные сущности нельзя left={left}, right={right}")
    
    def __ifloordiv__(self, other) -> Vec2:
        if isinstance(other, Vec2):
            return self.floordiv_vec_(<Vec2>other)
        elif isinstance(other, np.ndarray) or isinstance(other, tuple) or isinstance(other, list) or isinstance(other, memoryview):
            return self.floordiv_xy_(<real>(other[0]), <real>(other[1]))
        elif isinstance(other, int) or isinstance(other, float):
            return self.floordiv_num_(<real>other)
        else:
            raise NotImplementedError(f"Поделить на данную сущность нельзя  other={other}")


    @cython.cdivision(True)
    cpdef Vec2 mod_num_(self, real num):
        self.x %= num
        self.y %= num
        return self

    @cython.cdivision(True)
    cpdef Vec2 mod_num(self, real num):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mod_num_(num)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_vec_(self, Vec2 vec):
        self.x %= vec.x
        self.y %= vec.y
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_vec(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mod_vec_(vec)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_(self, Vec2 vec):
        self.x %= vec.x
        self.y %= vec.y
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod(self, Vec2 vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mod_vec_(vec)
        return result

    @cython.cdivision(True)
    cpdef Vec2 mod_xy_(self, real x, real y):
        self.x %= x
        self.y %= y
        return self

    @cython.cdivision(True)
    cpdef Vec2 mod_xy(self, real x, real y):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mod_xy_(x, y)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_tup_(self, tuple vec):
        self.x %= <real>(vec[0])
        self.y %= <real>(vec[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_tup(self, tuple vec):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mod_tup_(vec)
        return result   

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_list_(self, list tup):
        self.x %= <real>(tup[0])
        self.y %= <real>(tup[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_list(self, list tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mod_list_(tup)
        return result

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_arr_(self, rational[:] arr):
        self.x %= <real>(arr[0])
        self.y %= <real>(arr[1])
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Vec2 mod_arr(self, rational[:] tup):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.mod_arr_(tup)
        return result

    def __mod__(left, right):
        if isinstance(left, Vec2):
            if isinstance(right, Vec2):
                return (<Vec2>left).mod_vec(<Vec2>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview) :
                return (<Vec2>left).mod_xy(<real>(right[0]), <real>(right[1]))
            elif isinstance(right, int) or isinstance(right, float):
                return (<Vec2>left).mod_num(<real>right)

        elif isinstance(right, Vec2):
            if isinstance(left, np.ndarray) or isinstance(left, tuple) or isinstance(left, list) or isinstance(left, memoryview):
                return (Vec2(<real>(left[0]), <real>(left[1]))).mod_vec_(<Vec2>right)
            elif isinstance(left, int) or isinstance(left, float):
                return Vec2(<real>left, <real>left).mod_vec_(<Vec2>right)     
        raise NotImplementedError(f"Поделить данные сущности нельзя left={left}, right={right}")
    
    def __imod__(self, other) -> Vec2:
        if isinstance(other, Vec2):
            return self.mod_vec_(<Vec2>other)
        elif isinstance(other, np.ndarray) or isinstance(other, tuple) or isinstance(other, list) or isinstance(other, memoryview):
            return self.mod_xy_(<real>(other[0]), <real>(other[1]))
        elif isinstance(other, int) or isinstance(other, float):
            return self.mod_num_(<real>other)
        else:
            raise NotImplementedError(f"Поделить на данную сущность нельзя  other={other}")

    cpdef real len(self):
        return sqrt(self.x*self.x + self.y*self.y)

    cpdef real len_sqared(self):
        return self.x*self.x + self.y*self.y

    cpdef Vec2 abs_(self):
        self.x = fabs(self.x)
        self.y = fabs(self.y)

    cpdef Vec2 abs(self):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.abs_()
        return result

    def __abs__(self):
        return self.abs()

    @cython.cdivision(True)
    cpdef Vec2 norm_(self):
        cdef real length = self.len()
        self.x /= length
        self.y /= length

    @cython.cdivision(True)
    cpdef Vec2 norm(self):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.norm_()
        return result

    @cython.cdivision(True)
    cpdef Vec2 round_(self, int ndigits=0):
        if ndigits == 0:
            self.x = round(self.x)
            self.y = round(self.y)    
        else:
            self.mul_num_(10**ndigits)
            self.x = round(self.x)
            self.y = round(self.y)
            self.div_num_(10**ndigits)
        return self

    @cython.cdivision(True)
    cpdef Vec2 round(self, int ndigits=0):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.round_(ndigits)
        return result

    @cython.cdivision(True)
    cpdef Vec2 ceil_(self, int ndigits=0):
        if ndigits == 0:
            self.x = ceil(self.x)
            self.y = ceil(self.y)    
        else:
            self.mul_num_(10**ndigits)
            self.x = ceil(self.x)
            self.y = ceil(self.y)
            self.div_num_(10**ndigits)
        return self

    @cython.cdivision(True)
    cpdef Vec2 ceil(self, int ndigits=0):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.ceil_(ndigits)
        return result

    @cython.cdivision(True)
    cpdef Vec2 floor_(self, int ndigits=0):
        if ndigits == 0:
            self.x = floor(self.x)
            self.y = floor(self.y)    
        else:
            self.mul_num_(10**ndigits)
            self.x = floor(self.x)
            self.y = floor(self.y)
            self.div_num_(10**ndigits)
        return self

    @cython.cdivision(True)
    cpdef Vec2 floor(self, int ndigits=0):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.floor_(ndigits)
        return result

    @cython.cdivision(True)
    cpdef Vec2 trunc_(self, int ndigits=0):
        if ndigits == 0:
            self.x = trunc(self.x)
            self.y = trunc(self.y)    
        else:
            self.mul_num_(10**ndigits)
            self.x = trunc(self.x)
            self.y = trunc(self.y)
            self.div_num_(10**ndigits)
        return self

    @cython.cdivision(True)
    cpdef Vec2 trunc(self, int ndigits=0):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.trunc_(ndigits)
        return result

    @cython.nonecheck(False)
    cpdef real cross(self, Vec2 right):
        return self.x * right.y - self.y * right.x

    cpdef real cross_xy(self, real x, real y):
        return self.x * y - self.y * x

    def __and__(left, right):
        if isinstance(left, Vec2):
            if isinstance(right, Vec2):
                return (<Vec2>left).cross(<Vec2>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview) :
                return (<Vec2>left).cross_xy(<real>(right[0]), <real>(right[1]))

        elif isinstance(right, Vec2):
            if isinstance(left, np.ndarray) or isinstance(left, tuple) or isinstance(left, list) or isinstance(left, memoryview):
                return (Vec2(<real>(left[0]), <real>(left[1]))).cross(<Vec2>right)

        raise NotImplementedError(f"Векторно перемножить нельзя left={left}, right={right}")

    @cython.cdivision(True)
    cpdef real angle_to_xy(self, real x, real y, int degrees=0):
        cdef real angle = atan2(y, x) - atan2(self.y, self.x)
        if angle > pi:
            angle -= 2*pi
        elif angle <= -pi:
            angle += 2*pi
        if degrees != 0:
            angle *= 180.0/pi
        return angle

    @cython.nonecheck(False)
    cpdef real angle_to(self, Vec2 vec, int degrees=0):
        return self.angle_to_xy(vec.x, vec.y, degrees)

    cpdef Vec2 rotate90_(self):
        cdef real buf = self.x
        self.x = -self.y
        self.y = buf
        return self

    cpdef Vec2 rotate90(self):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.rotate90_()
        return result

    cpdef Vec2 rotate_minus90_(self):
        cdef real buf = self.x
        self.x = self.y
        self.y = -buf
        return self

    cpdef Vec2 rotate_minus90(self):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.rotate_minus90_()
        return result

    @cython.cdivision(True)
    cpdef Vec2 rotate_(self, real angle, int degrees=0):
        if degrees != 0:
            angle /= 180.0/pi
        cdef real s = sin(angle)
        cdef real c = cos(angle)
        
        cdef real x = self.x * c - self.y * s
        self.y = self.x * s + self.y * c
        self.x = x
        return self

    cpdef Vec2 rotate(self, real angle, int degrees=0):
        cdef Vec2 result = Vec2(self.x, self.y)
        result.rotate_(angle, degrees)
        return result

cdef class Vec3:
    def __cinit__(self, x: real, y: real, z: real):
        self.x = x
        self.y = y
        self.z = z 

    def __str__(self):
        return f'({self.x:.2f}, {self.y:.2f}, {self.z:.2f})'

    def __repr__(self):
        return f'Vec3({self.x}, {self.y}, {self.z})'