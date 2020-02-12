from libc.math cimport sin, cos, pi, fabs
import numpy as np
from .vectors cimport CMP_TOL
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_GE, Py_GT, Py_NE
cimport cython



cdef class Mat2:
    @classmethod
    def from_xaxis(cls, xaxis):
        if not isinstance(xaxis, Vec2):
            if isinstance(xaxis, dict):
                xaxis = Vec2.from_dict(xaxis)
            else:
                xaxis = Vec2.from_list(xaxis)
        xaxis.norm_()
        cdef Vec2 yaxis = xaxis.rotate90()
        return cls(xaxis, yaxis)

    @classmethod
    def from_yaxis(cls, yaxis):
        if not isinstance(yaxis, Vec2):
            if isinstance(yaxis, dict):
                yaxis = Vec2.from_dict(yaxis)
            else:
                yaxis = Vec2.from_list(yaxis)
        yaxis.norm_()
        cdef Vec2 xaxis = yaxis.rotate_minus90()
        return cls(xaxis, yaxis)

    @classmethod
    def from_angle(cls, angle, degrees=0):
        if degrees:
            angle /= 180/pi
        cdef real s = sin(angle)
        cdef real c = cos(angle)    
        return cls([c, s], [-s, c])

    @classmethod
    def eye(cls):
        return cls(1,0,0,1)

    @classmethod
    def zeros(cls):
        return cls(0,0,0,0)

    def __cinit__(self, *args):
        cdef int alen = len(args)
        cdef int alen2
        
        if alen == 2:
        # 2 вектора или 2 списка/кортежа
            self.m11 = args[0][0]
            self.m12 = args[0][1]
            self.m21 = args[1][0]
            self.m22 = args[1][1]

        elif alen == 1:
            if isinstance(args[0], Mat2):
                self.m11 = args[0].m11
                self.m12 = args[0].m12
                self.m21 = args[0].m21
                self.m22 = args[0].m22
            else:
                alen2 = len(args[0])
                if alen2 == 2:
                # 1 список списков
                    self.m11 = args[0][0][0]
                    self.m12 = args[0][0][1]
                    self.m21 = args[0][1][0]
                    self.m22 = args[0][1][1]
                elif alen2 == 4:
                # 1 список
                    self.m11 = args[0][0]
                    self.m12 = args[0][1]
                    self.m21 = args[0][2]
                    self.m22 = args[0][3]
                else:
                    raise ValueError(f'Невозможно создать экземпляр Mat2 из параметров {args}')
        elif alen == 4:
        # просто 4 числа
            self.m11 = args[0]
            self.m12 = args[1]
            self.m21 = args[2]
            self.m22 = args[3]  
        elif alen == 0:
            self.m11 = 0.0
            self.m12 = 0.0
            self.m21 = 0.0
            self.m22 = 0.0
        else:                        
            raise ValueError(f'Невозможно создать экземпляр Mat2 из параметров {args}')

    
    cpdef Mat2 copy(self):
        return Mat2(self.m11, self.m12, self.m21, self.m22)

    cpdef Mat2 clone(self):
        return Mat2(self.m11, self.m12, self.m21, self.m22)

    def __str__(self):
        return f'[[{self.m11:.2f}, {self.m12:.2f}], [{self.m21:.2f}, {self.m22:.2f}]]'

    def __repr__(self):
        return f'Mat2([[{self.m11}, {self.m12}], [{self.m21}, {self.m22}]])'

    cpdef real[:,:] as_np(self):
        return np.array([[self.m11, self.m12], [self.m21, self.m22]])

    cpdef Vec2 i_axis(self):
        return Vec2(self.m11, self.m12)

    cpdef Vec2 j_axis(self):
        return Vec2(self.m21, self.m22)

    cpdef Vec2 x_axis(self):
        return Vec2(self.m11, self.m12)

    cpdef Vec2 y_axis(self):
        return Vec2(self.m21, self.m22)

    cpdef Mat2 transpose(self):
        return Mat2(self.m11, self.m21, self.m12, self.m22)
    
    @property
    def T(self) -> Mat2:
        return self.transpose()

    cpdef real det(self):
        return self.m11*self.m22 - self.m12*self.m21

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 inverse(self):
        cdef real det = self.det()
        if fabs(det) - 1.0 < CMP_TOL:
            return self.transpose()
        if fabs(det) < CMP_TOL:
            return Mat2()
        return Mat2(
             self.m22 / det, -self.m12 / det,
            -self.m21 / det,  self.m11 / det
        )

    @property
    def _1(self) -> Mat2:
        return self.inverse()

    @cython.nonecheck(False)
    cpdef Mat2 mul_mat_elements_(self, Mat2 right):
        self.m11 *= right.m11
        self.m12 *= right.m12
        self.m21 *= right.m21
        self.m22 *= right.m22
        return self

    @cython.nonecheck(False)
    cpdef Mat2 mul_mat_elements(self, Mat2 right):
        return Mat2(
            self.m11 * right.m11,
            self.m12 * right.m12,
            self.m21 * right.m21,
            self.m22 * right.m22 
        )

    @cython.nonecheck(False)
    cpdef Mat2 mul_mat_(self, Mat2 right):
        cdef real m11 = self.m11 * right.m11 + self.m12 * right.m21
        cdef real m12 = self.m11 * right.m12 + self.m12 * right.m22
        cdef real m21 = self.m11 * right.m12 + self.m12 * right.m22
        cdef real m22 = self.m21 * right.m12 + self.m22 * right.m22
        self.m11 = m11
        self.m12 = m12
        self.m21 = m21
        self.m22 = m22
        return self

    @cython.nonecheck(False)
    cpdef Mat2 mul_mat(self, Mat2 right):
        return Mat2(
            self.m11 * right.m11 + self.m12 * right.m21,
            self.m11 * right.m12 + self.m12 * right.m22,
            self.m11 * right.m12 + self.m12 * right.m22,
            self.m21 * right.m12 + self.m22 * right.m22
        )
    
    @cython.nonecheck(False)
    cpdef Vec2 mul_vec(self, Vec2 vec):
        return Vec2(
            self.m11 * vec.x + self.m12 * vec.y,
			self.m21 * vec.x + self.m22 * vec.y
        )

    @cython.nonecheck(False)
    cpdef Mat2 mul_num_(self, real num):
        self.m11 *= num
        self.m12 *= num
        self.m21 *= num
        self.m22 *= num
        return self

    @cython.nonecheck(False)
    cpdef Mat2 mul_num(self, real num):
        return Mat2(
            self.m11 * num,
            self.m12 * num,
            self.m21 * num,
            self.m22 * num
        )

    def __mul__(left, right):
        if isinstance(left, Mat2):
            if isinstance(right, Vec2):
                return (<Mat2>left).mul_vec(<Vec2>right)
            elif isinstance(right, Mat2):
                return (<Mat2>left).mul_mat(<Mat2>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview):
                return (<Mat2>left).mul_vec(Vec2(right[0], right[1]))
            elif isinstance(right, int) or isinstance(right, float):
                return (<Mat2>left).mul_num(<real>right)
        elif isinstance(left, int) or isinstance(left, float):
            return (<Mat2>right).mul_num(<real>left)
        raise NotImplementedError(f"Перемножить данные сущности нельзя left={left}, right={right}")
    

    def __imul__(self, right):
        if isinstance(right, Mat2):
            return self.mul_mat_(<Mat2>right)
        elif isinstance(right, int) or isinstance(right, float):
            return self.mul_num_(<real>right)
        raise NotImplementedError(f"Перемножить данные сущности нельзя left={self}, right={right}")
    
    @cython.nonecheck(False)
    cpdef Mat2 add_num_(self, real num):
        self.m11 += num
        self.m12 += num
        self.m21 += num
        self.m22 += num
        return self

    @cython.nonecheck(False)
    cpdef Mat2 add_num(self, real num):
        return Mat2(
            self.m11 + num,
            self.m12 + num,
            self.m21 + num,
            self.m22 + num
        )


    @cython.nonecheck(False)
    cpdef Mat2 add_mat_(self, Mat2 mat):
        self.m11 += mat.m11
        self.m12 += mat.m12 
        self.m21 += mat.m21
        self.m22 += mat.m22
        return self

    @cython.nonecheck(False)
    cpdef Mat2 add_mat(self, Mat2 mat):
        return Mat2(
            self.m11 + mat.m11,
            self.m12 + mat.m12, 
            self.m21 + mat.m21,
            self.m22 + mat.m22
        )

    def __add__(left, right):
        if isinstance(left, Mat2):
            if isinstance(right, Mat2):
                return (<Mat2>left).add_mat(<Mat2>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Mat2>left).add_num(<real>right)
        elif isinstance(left, int) or isinstance(left, float):
            return (<Mat2>right).add_num(<real>left)
        raise NotImplementedError(f"Сложить данные сущности нельзя left={left}, right={right}")
    
    def __iadd__(self, right):
        if isinstance(right, Mat2):
            return self.add_mat_(<Mat2>right)
        elif isinstance(right, int) or isinstance(right, float):
            return self.add_num_(<real>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя self={self}, right={right}")
  
    @cython.nonecheck(False)
    cpdef Mat2 neg_(self):
        self.m11 = -self.m11
        self.m12 = -self.m12 
        self.m21 = -self.m21
        self.m22 = -self.m22
        return self

    @cython.nonecheck(False)
    cpdef Mat2 neg(self):
        return Mat2(
            -self.m11,
            -self.m12, 
            -self.m21,
            -self.m22
        )

    def __neg__(self):
        return self.neg()

    @cython.nonecheck(False)
    def __richcmp__(v1, v2, int op):
        if op == Py_EQ:
            return  fabs(v1[0][0] - v2[0][0]) < CMP_TOL and fabs(v1[0][1] - v2[0][1]) < CMP_TOL  \
                and fabs(v1[1][0] - v2[1][0]) < CMP_TOL and fabs(v1[1][1] - v2[1][1]) < CMP_TOL
        elif op == Py_NE:
            return fabs(v1[0][0] - v2[0][0]) >= CMP_TOL or fabs(v1[0][1] - v2[0][1]) >= CMP_TOL  \
                or fabs(v1[1][0] - v2[1][0]) >= CMP_TOL or fabs(v1[1][1] - v2[1][1]) >= CMP_TOL
        raise NotImplementedError("Такой тип сравнения не поддерживается")

    def __getitem__(self, key):
        if key == 0:
            return self.x_axis()
        if key == 1:
            return self.y_axis()
        if isinstance(key, str):
            if key == 'm11':
                return self.m11
            if key == 'm12':
                return self.m12
            if key == 'm21':
                return self.m21
            if key == 'm22':
                return self.m22
        elif isinstance(key, tuple) and (len(key)==2):
            if key[0] == 0 and key[1] == 0:
                return self.m11
            if key[0] == 0 and key[1] == 1:
                return self.m12
            if key[0] == 1 and key[1] == 0:
                return self.m21
            if key[0] == 1 and key[1] == 1:
                return self.m22
        raise KeyError(f'Неправильный индекс {key}')

    cpdef list keys(self):
        return ['m11', 'm12', 'm21', 'm22']

    def __iter__(self):
        return self.get_generator()

    def get_generator(self):
        yield (self.m11, self.m12) 
        yield (self.m21, self.m22)   

    
    cpdef tuple as_tuple(self):
        return ((self.m11, self.m12), (self.m21, self.m22))     
    
    @cython.nonecheck(False)
    cpdef Mat2 sub_num_(self, real num):
        self.m11 -= num
        self.m12 -= num
        self.m21 -= num
        self.m22 -= num
        return self

    @cython.nonecheck(False)
    cpdef Mat2 sub_num(self, real num):
        return Mat2(
            self.m11 - num,
            self.m12 - num,
            self.m21 - num,
            self.m22 - num
        )


    @cython.nonecheck(False)
    cpdef Mat2 sub_mat_(self, Mat2 mat):
        self.m11 -= mat.m11
        self.m12 -= mat.m12 
        self.m21 -= mat.m21
        self.m22 -= mat.m22
        return self

    @cython.nonecheck(False)
    cpdef Mat2 sub_mat(self, Mat2 mat):
        return Mat2(
            self.m11 - mat.m11,
            self.m12 - mat.m12, 
            self.m21 - mat.m21,
            self.m22 - mat.m22
        )   

    def __sub__(left, right):
        if isinstance(left, Mat2):
            if isinstance(right, Mat2):
                return (<Mat2>left).sub_mat(<Mat2>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Mat2>left).sub_num(<real>right)
        elif isinstance(left, int) or isinstance(left, float):
            return ((<Mat2>right).sub_num(<real>left)).neg()
        raise NotImplementedError(f"Вычесть данные сущности нельзя left={left}, right={right}")
    
    def __isub__(self, right):
        if isinstance(right, Mat2):
            return self.sub_mat_(<Mat2>right)
        elif isinstance(right, int) or isinstance(right, float):
            return self.sub_num_(<real>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя self={self}, right={right}")

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 div_num_(self, real num):
        self.m11 /= num
        self.m12 /= num
        self.m21 /= num
        self.m22 /= num
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 div_num(self, real num):
        return Mat2(
            self.m11 / num,
            self.m12 / num,
            self.m21 / num,
            self.m22 / num
        )


    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 div_mat_(self, Mat2 mat):
        self.m11 /= mat.m11
        self.m12 /= mat.m12 
        self.m21 /= mat.m21
        self.m22 /= mat.m22
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 div_mat(self, Mat2 mat):
        return Mat2(
            self.m11 / mat.m11,
            self.m12 / mat.m12, 
            self.m21 / mat.m21,
            self.m22 / mat.m22
        )   

    def __truediv__(left, right):
        if isinstance(left, Mat2):
            if isinstance(right, Mat2):
                return (<Mat2>left).div_mat(<Mat2>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Mat2>left).div_num(<real>right)
        raise NotImplementedError(f"Вычесть данные сущности нельзя left={left}, right={right}")
    
    def __itruediv__(self, right):
        if isinstance(right, Mat2):
            return self.div_mat_(<Mat2>right)
        elif isinstance(right, int) or isinstance(right, float):
            return self.div_num_(<real>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя self={self}, right={right}")
  