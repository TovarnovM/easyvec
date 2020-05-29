from libc.math cimport sin, cos, pi, fabs, sqrt, acos
import numpy as np
from .vectors cimport CMP_TOL
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_GE, Py_GT, Py_NE
cimport cython

cpdef Vec3 _convert(object candidate):
    if isinstance(candidate, Vec3):
        return <Vec3>(candidate)
    else:
        return Vec3(candidate[0], candidate[1], candidate[2])

@cython.final
cdef class Mat2:
    """
    Класс представляющий матрицу 2х2
    Служит для афинных преобразований векторов Vec2 

    Поля:
        |m11 m12|
        |m21 m22|

    Примеры создания матрицы
        |1 2|
        |3 4|
        >>> Mat2(1,2,3,4)
        >>> Mat2((1,2),(3,4))
        >>> Mat2(Vec2(1,2), Vec2(3,4))
        >>> Mat2(Vec2(1,2),(3,4))

    Пример создания матрицы
        |0 0|
        |0 0|
    >>> Mat2(0,0,0,0)
    >>> Mat2.zeros()
    >>> Mat2()

    Пример создания матрицы
        |1 0|
        |0 1|
    >>> Mat2(1,0,0,1)
    >>> Mat2.eye()
    """
    @classmethod
    def from_xaxis(cls, xaxis):
        """
        Создать матрицу повернутой системы координат, ось Ox которой имеет координаты xaxis в глобальной
        """
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
        """
        Создать матрицу повернутой системы координат, ось Oy которой имеет координаты yaxis в глобальной
        """
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
        """
        Создать матрицу повернутой системы координат, оси которой поверуты на угол angle
        """
        if degrees:
            angle /= 180/pi
        cdef real s = sin(angle)
        cdef real c = cos(angle)    
        return cls([c, s], [-s, c])

    @classmethod
    def eye(cls):
        """
        Создать единичную матрицу
        |1 0|
        |0 1|
        """
        return cls(1,0,0,1)

    @classmethod
    def zeros(cls):
        """
        Создать нулевую матрицу
        |0 0|
        |0 0|
        """
        return cls(0,0,0,0)

    def __cinit__(self, *args):
        """
        Конструктор
        """
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
        """
        return np.array([[self.m11, self.m12], [self.m21, self.m22]])
        """
        return np.array([[self.m11, self.m12], [self.m21, self.m22]])

    cpdef Vec2 i_axis(self):
        """
        return Vec2(self.m11, self.m12)
        """
        return Vec2(self.m11, self.m12)

    cpdef Vec2 j_axis(self):
        """
        return Vec2(self.m21, self.m22)
        """
        return Vec2(self.m21, self.m22)

    cpdef Vec2 x_axis(self):
        """
        return Vec2(self.m11, self.m12)
        """
        return Vec2(self.m11, self.m12)

    cpdef Vec2 y_axis(self):
        """
        return Vec2(self.m21, self.m22)
        """
        return Vec2(self.m21, self.m22)

    cpdef Mat2 transpose(self):
        """
        Возвращает транспонированную матрицу
        """
        return Mat2(self.m11, self.m21, self.m12, self.m22)
    
    @property
    def T(self) -> Mat2:
        """
        Возвращает транспонированную матрицу
        """
        return self.transpose()

    cpdef real det(self):
        """
        Возвращает определитель матрицы
        """
        return self.m11*self.m22 - self.m12*self.m21

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 inverse(self):
        """
        Возвращает обратную матрицу
        """
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
        """
        Возвращает обратную матрицу
        """
        return self.inverse()

    @cython.nonecheck(False)
    cpdef Mat2 mul_mat_elements_(self, Mat2 right):
        """
        Изменяет матрицу. Поэлементно перемножает с другой матрицей
        """
        self.m11 *= right.m11
        self.m12 *= right.m12
        self.m21 *= right.m21
        self.m22 *= right.m22
        return self

    @cython.nonecheck(False)
    cpdef Mat2 mul_mat_elements(self, Mat2 right):
        """
        Возвращает матрицу с поэлементно перемножиными компонентами
        """
        return Mat2(
            self.m11 * right.m11,
            self.m12 * right.m12,
            self.m21 * right.m21,
            self.m22 * right.m22 
        )

    @cython.nonecheck(False)
    cpdef Mat2 mul_mat_(self, Mat2 right):
        """
        Изменяет матрицу. Производит матричное произведение на другую матрицу

        Mat2(
            self.m11 * right.m11 + self.m12 * right.m21,
            self.m11 * right.m12 + self.m12 * right.m22,
            self.m11 * right.m12 + self.m12 * right.m22,
            self.m21 * right.m12 + self.m22 * right.m22
        )
        """
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
        """
        Возвращает матричное произведение с другой матрицей

        return Mat2(
            self.m11 * right.m11 + self.m12 * right.m21,
            self.m11 * right.m12 + self.m12 * right.m22,
            self.m11 * right.m12 + self.m12 * right.m22,
            self.m21 * right.m12 + self.m22 * right.m22
        )
        """
        return Mat2(
            self.m11 * right.m11 + self.m12 * right.m21,
            self.m11 * right.m12 + self.m12 * right.m22,
            self.m11 * right.m12 + self.m12 * right.m22,
            self.m21 * right.m12 + self.m22 * right.m22
        )
    
    @cython.nonecheck(False)
    cpdef Vec2 mul_vec(self, Vec2 vec):
        """
        Возвращает матричное произведение с другим вектором

        return Vec2(
            self.m11 * vec.x + self.m12 * vec.y,
			self.m21 * vec.x + self.m22 * vec.y
        )
        """
        return Vec2(
            self.m11 * vec.x + self.m12 * vec.y,
			self.m21 * vec.x + self.m22 * vec.y
        )

    @cython.nonecheck(False)
    cpdef Mat2 mul_num_(self, real num):
        """
        Изменяет матрицу. Производит поэлементное произведение на число

        self.m11 *= num
        self.m12 *= num
        self.m21 *= num
        self.m22 *= num
        """
        self.m11 *= num
        self.m12 *= num
        self.m21 *= num
        self.m22 *= num
        return self

    @cython.nonecheck(False)
    cpdef Mat2 mul_num(self, real num):
        """
        Возвращает матрицу с элементами, умножиными на число

        return Mat2(
            self.m11 * num,
            self.m12 * num,
            self.m21 * num,
            self.m22 * num
        )
        """    
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
        """
        self.m11 += num
        self.m12 += num
        self.m21 += num
        self.m22 += num
        """
        self.m11 += num
        self.m12 += num
        self.m21 += num
        self.m22 += num
        return self

    @cython.nonecheck(False)
    cpdef Mat2 add_num(self, real num):
        """
        return Mat2(
            self.m11 + num,
            self.m12 + num,
            self.m21 + num,
            self.m22 + num
        )
        """
        return Mat2(
            self.m11 + num,
            self.m12 + num,
            self.m21 + num,
            self.m22 + num
        )


    @cython.nonecheck(False)
    cpdef Mat2 add_mat_(self, Mat2 mat):
        """
        self.m11 += mat.m11
        self.m12 += mat.m12 
        self.m21 += mat.m21
        self.m22 += mat.m22
        """
        self.m11 += mat.m11
        self.m12 += mat.m12 
        self.m21 += mat.m21
        self.m22 += mat.m22
        return self

    @cython.nonecheck(False)
    cpdef Mat2 add_mat(self, Mat2 mat):
        """
        return Mat2(
            self.m11 + mat.m11,
            self.m12 + mat.m12, 
            self.m21 + mat.m21,
            self.m22 + mat.m22
        )
        """
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
        """
        self.m11 = -self.m11
        self.m12 = -self.m12 
        self.m21 = -self.m21
        self.m22 = -self.m22
        """
        self.m11 = -self.m11
        self.m12 = -self.m12 
        self.m21 = -self.m21
        self.m22 = -self.m22
        return self

    @cython.nonecheck(False)
    cpdef Mat2 neg(self):
        """
        return Mat2(
            -self.m11,
            -self.m12, 
            -self.m21,
            -self.m22
        )
        """
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
        """
        return ['m11', 'm12', 'm21', 'm22']
        """
        return ['m11', 'm12', 'm21', 'm22']

    def __iter__(self):
        return self.get_generator()

    def get_generator(self):
        yield (self.m11, self.m12) 
        yield (self.m21, self.m22)   

    
    cpdef tuple as_tuple(self):
        """
        return ((self.m11, self.m12), (self.m21, self.m22)) 
        """
        return ((self.m11, self.m12), (self.m21, self.m22))     
    
    @cython.nonecheck(False)
    cpdef Mat2 sub_num_(self, real num):
        """
        self.m11 -= num
        self.m12 -= num
        self.m21 -= num
        self.m22 -= num
        """
        self.m11 -= num
        self.m12 -= num
        self.m21 -= num
        self.m22 -= num
        return self

    @cython.nonecheck(False)
    cpdef Mat2 sub_num(self, real num):
        """
        return Mat2(
            self.m11 - num,
            self.m12 - num,
            self.m21 - num,
            self.m22 - num
        )
        """
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
        """
        return Mat2(
            self.m11 - mat.m11,
            self.m12 - mat.m12, 
            self.m21 - mat.m21,
            self.m22 - mat.m22
        )  
        """
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
        """
        self.m11 /= num
        self.m12 /= num
        self.m21 /= num
        self.m22 /= num
        """
        self.m11 /= num
        self.m12 /= num
        self.m21 /= num
        self.m22 /= num
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 div_num(self, real num):
        """
        return Mat2(
            self.m11 / num,
            self.m12 / num,
            self.m21 / num,
            self.m22 / num
        )
        """
        return Mat2(
            self.m11 / num,
            self.m12 / num,
            self.m21 / num,
            self.m22 / num
        )


    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 div_mat_(self, Mat2 mat):
        """
        self.m11 /= mat.m11
        self.m12 /= mat.m12 
        self.m21 /= mat.m21
        self.m22 /= mat.m22
        """
        self.m11 /= mat.m11
        self.m12 /= mat.m12 
        self.m21 /= mat.m21
        self.m22 /= mat.m22
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat2 div_mat(self, Mat2 mat):
        """
        return Mat2(
            self.m11 / mat.m11,
            self.m12 / mat.m12, 
            self.m21 / mat.m21,
            self.m22 / mat.m22
        )  
        """
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

@cython.final
cdef class Mat3:
    @classmethod
    def eye(cls):
        return cls(1, 0, 0, \
                   0, 1, 0, \
                   0, 0, 1  )

    @classmethod
    def zeros(cls):
        return cls(0, 0, 0, \
                   0, 0, 0, \
                   0, 0, 0  )

    def __cinit__(self, m11: real, m12: real, m13: real,  
                        m21: real, m22: real, m23: real, 
                        m31: real, m32: real, m33: real):
        self.m11 = m11
        self.m12 = m12
        self.m13 = m13
        self.m21 = m21
        self.m22 = m22
        self.m23 = m23
        self.m31 = m31
        self.m32 = m32
        self.m33 = m33

    cpdef Mat3 copy(self):
        return Mat3(self.m11, self.m12, self.m13,  
                    self.m21, self.m22, self.m23, 
                    self.m31, self.m32, self.m33)

    cpdef Mat3 clone(self):
        return Mat3(self.m11, self.m12, self.m13, 
                    self.m21, self.m22, self.m23, 
                    self.m31, self.m32, self.m33)

    def __str__(self):
        return f'''[[{self.m11:<7.3f}, {self.m12:<7.3f}, {self.m13:<7.3f}], 
 [{self.m21:<7.3f}, {self.m22:<7.3f}, {self.m23:<7.3f}],
 [{self.m31:<7.3f}, {self.m32:<7.3f}, {self.m33:<7.3f}]]'''

    def __repr__(self):
        return f'''Mat3({self.m11:<7.3f}, {self.m12:<7.3f}, {self.m13:<7.3f}, 
     {self.m21:<7.3f}, {self.m22:<7.3f}, {self.m23:<7.3f},
     {self.m31:<7.3f}, {self.m32:<7.3f}, {self.m33:<7.3f})'''

    cpdef real[:,:] as_np(self):
        return np.array([[self.m11, self.m12, self.m13],[self.m21, self.m22, self.m23],[self.m31, self.m32, self.m33]])

    cpdef list as_list(self):
        return [self.m11, self.m12, self.m13, self.m21, self.m22, self.m23, self.m31, self.m32, self.m33]

    cpdef tuple as_tuple(self):
        return ((self.m11, self.m12, self.m13), 
        (self.m21, self.m22, self.m23), 
        (self.m31, self.m32, self.m33))

    cpdef Vec3 i_axis(self):
        return Vec3(self.m11, self.m12, self.m13)

    cpdef Vec3 j_axis(self):
        return Vec3(self.m21, self.m22, self.m23)

    cpdef Vec3 k_axis(self):
        return Vec3(self.m31, self.m32, self.m33)    
    
    cpdef Vec3 x_axis(self):
        return Vec3(self.m11, self.m12, self.m13)

    cpdef Vec3 y_axis(self):
        return Vec3(self.m21, self.m22, self.m23)

    cpdef Vec3 z_axis(self):
        return Vec3(self.m31, self.m32, self.m33)    

    cpdef Mat3 transpose(self):
        return Mat2(self.m11, self.m21, self.m31, \
                    self.m12, self.m22, self.m32, \
                    self.m13, self.m23, self.m33)

    cpdef Mat3 transpose_(self):
        self.m12, self.m21 = self.m21, self.m12
        self.m13, self.m31 = self.m31, self.m13
        self.m23, self.m32 = self.m32, self.m23

    
    @property
    def T(self) -> Mat3:
        return self.transpose()   

    cpdef real det(self):
        return  self.m11 * self.m22 * self.m33 + \
                self.m12 * self.m23 * self.m31 + \
                self.m13 * self.m21 * self.m32 - \
                self.m13 * self.m22 * self.m31 - \
                self.m11 * self.m23 * self.m32 - \
                self.m12 * self.m21 * self.m33


    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat3 inverse(self):
        cdef real det = self.det()
        if fabs(det) - 1 < CMP_TOL:
            return self.transpose()
        if fabs(det) < CMP_TOL:
            return Mat3(1,0,0, 0,1,0, 0,0,1)
        return Mat3(
            (self.m22 * self.m33 - self.m32 * self.m23) / det,
            (self.m13 * self.m32 - self.m12 * self.m33) / det,
            (self.m12 * self.m23 - self.m13 * self.m22) / det,
            (self.m23 * self.m31 - self.m21 * self.m33) / det,
            (self.m11 * self.m33 - self.m13 * self.m31) / det,
            (self.m21 * self.m13 - self.m11 * self.m23) / det,
            (self.m21 * self.m32 - self.m31 * self.m22) / det,
            (self.m31 * self.m12 - self.m11 * self.m32) / det,
            (self.m11 * self.m22 - self.m21 * self.m12) / det
        )

    @property
    def _1(self) -> Mat3:
        return self.inverse()

    @cython.nonecheck(False)
    cpdef Mat3 mul_mat_elements_(self, Mat3 right):
        self.m11 *= right.m11
        self.m12 *= right.m12
        self.m13 *= right.m13
        self.m21 *= right.m21
        self.m22 *= right.m22
        self.m23 *= right.m23
        self.m31 *= right.m31
        self.m32 *= right.m32
        self.m33 *= right.m33
        return self

    @cython.nonecheck(False)
    cpdef Mat3 mul_mat_elements(self, Mat3 right):
        cdef Mat3 res = self.copy()
        return res.mul_mat_elements_(right)

    @cython.nonecheck(False)
    cpdef Mat3 mul_mat_(self, Mat3 b):
        cdef:
            real m11 = self.m11 * b.m11 + self.m12 * b.m21 + self.m13 * b.m31
            real m12 = self.m11 * b.m12 + self.m12 * b.m22 + self.m13 * b.m32
            real m13 = self.m11 * b.m13 + self.m12 * b.m23 + self.m13 * b.m33

            real m21 = self.m21 * b.m11 + self.m22 * b.m21 + self.m23 * b.m31
            real m22 = self.m21 * b.m12 + self.m22 * b.m22 + self.m23 * b.m32
            real m23 = self.m21 * b.m13 + self.m22 * b.m23 + self.m23 * b.m33

            real m31 = self.m31 * b.m11 + self.m32 * b.m21 + self.m33 * b.m31
            real m32 = self.m31 * b.m12 + self.m32 * b.m22 + self.m33 * b.m32 
            real m33 = self.m31 * b.m13 + self.m32 * b.m23 + self.m33 * b.m33        
        self.m11 = m11
        self.m12 = m12
        self.m13 = m13
        self.m21 = m21
        self.m22 = m22
        self.m23 = m23
        self.m31 = m31
        self.m32 = m32
        self.m33 = m33

    @cython.nonecheck(False)
    cpdef Mat3 mul_mat(self, Mat3 b):
        return Mat3(self.m11 * b.m11 + self.m12 * b.m21 + self.m13 * b.m31,
            self.m11 * b.m12 + self.m12 * b.m22 + self.m13 * b.m32,
            self.m11 * b.m13 + self.m12 * b.m23 + self.m13 * b.m33,

            self.m21 * b.m11 + self.m22 * b.m21 + self.m23 * b.m31,
            self.m21 * b.m12 + self.m22 * b.m22 + self.m23 * b.m32,
            self.m21 * b.m13 + self.m22 * b.m23 + self.m23 * b.m33,

            self.m31 * b.m11 + self.m32 * b.m21 + self.m33 * b.m31,
            self.m31 * b.m12 + self.m32 * b.m22 + self.m33 * b.m32,
            self.m31 * b.m13 + self.m32 * b.m23 + self.m33 * b.m33)
    
    @cython.nonecheck(False)
    cpdef Vec3 mul_vec(self, Vec3 vec):
        return Vec3(
            self.m11 * vec.x + self.m12 * vec.y + self.m13 * vec.z,
			self.m21 * vec.x + self.m22 * vec.y + self.m23 * vec.z,
            self.m31 * vec.x + self.m32 * vec.y + self.m33 * vec.z
        )

    @cython.nonecheck(False)
    cpdef Mat3 mul_num_(self, real num):
        self.m11 *= num
        self.m12 *= num
        self.m13 *= num
        self.m21 *= num
        self.m22 *= num
        self.m23 *= num
        self.m31 *= num
        self.m32 *= num
        self.m33 *= num
        return self

    @cython.nonecheck(False)
    cpdef Mat3 mul_num(self, real num):
        return Mat3(
            self.m11 * num,
            self.m12 * num,
            self.m13 * num,
            self.m21 * num,
            self.m22 * num,
            self.m23 * num,
            self.m31 * num,
            self.m32 * num,
            self.m33 * num
        )

    def __mul__(left, right):
        if isinstance(left, Mat3):
            if isinstance(right, Vec3):
                return (<Mat3>left).mul_vec(<Vec3>right)
            elif isinstance(right, Mat3):
                return (<Mat3>left).mul_mat(<Mat3>right)
            elif isinstance(right, np.ndarray) or isinstance(right, tuple) or isinstance(right, list) or isinstance(right, memoryview):
                return (<Mat3>left).mul_vec(Vec3(right[0], right[1], right[2]))
            elif isinstance(right, int) or isinstance(right, float):
                return (<Mat3>left).mul_num(<real>right)
        elif isinstance(left, int) or isinstance(left, float):
            return (<Mat3>right).mul_num(<real>left)
        raise NotImplementedError(f"Перемножить данные сущности нельзя left={left}, right={right}")
    

    def __imul__(self, right):
        if isinstance(right, Mat3):
            return self.mul_mat_(<Mat3>right)
        elif isinstance(right, int) or isinstance(right, float):
            return self.mul_num_(<real>right)
        raise NotImplementedError(f"Перемножить данные сущности нельзя left={self}, right={right}")
        
    @cython.nonecheck(False)
    cpdef Mat3 add_num_(self, real num):
        self.m11 += num
        self.m12 += num
        self.m13 += num
        self.m21 += num
        self.m22 += num
        self.m23 += num
        self.m31 += num
        self.m32 += num
        self.m33 += num
        return self

    @cython.nonecheck(False)
    cpdef Mat3 add_num(self, real num):
        return Mat2(
            self.m11 + num,
            self.m12 + num,
            self.m13 + num,
            self.m21 + num,
            self.m22 + num,
            self.m23 + num,
            self.m31 + num,
            self.m32 + num,
            self.m33 + num
        )


    @cython.nonecheck(False)
    cpdef Mat3 add_mat_(self, Mat3 mat):
        self.m11 += mat.m11
        self.m12 += mat.m12 
        self.m13 += mat.m13 
        self.m21 += mat.m21
        self.m22 += mat.m22
        self.m23 += mat.m23
        self.m31 += mat.m31
        self.m32 += mat.m32
        self.m33 += mat.m33
        return self

    @cython.nonecheck(False)
    cpdef Mat3 add_mat(self, Mat3 mat):
        return Mat3(
            self.m11 + mat.m11,
            self.m12 + mat.m12, 
            self.m13 + mat.m13,
            self.m21 + mat.m21,
            self.m22 + mat.m22,
            self.m23 + mat.m23,
            self.m31 + mat.m31,
            self.m32 + mat.m32,
            self.m33 + mat.m33
        )

    def __add__(left, right):
        if isinstance(left, Mat3):
            if isinstance(right, Mat3):
                return (<Mat3>left).add_mat(<Mat3>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Mat3>left).add_num(<real>right)
        elif isinstance(left, int) or isinstance(left, float):
            return (<Mat3>right).add_num(<real>left)
        raise NotImplementedError(f"Сложить данные сущности нельзя left={left}, right={right}")
    
    def __iadd__(self, right):
        if isinstance(right, Mat3):
            return self.add_mat_(<Mat3>right)
        elif isinstance(right, int) or isinstance(right, float):
            return self.add_num_(<real>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя self={self}, right={right}")
  
    @cython.nonecheck(False)
    cpdef Mat3 neg_(self):
        self.m11 = -self.m11
        self.m12 = -self.m12 
        self.m13 = -self.m13
        self.m21 = -self.m21
        self.m22 = -self.m22
        self.m23 = -self.m23
        self.m31 = -self.m31
        self.m32 = -self.m32
        self.m33 = -self.m33
        return self

    @cython.nonecheck(False)
    cpdef Mat3 neg(self):
        return Mat2(
            -self.m11,
            -self.m12, 
            -self.m13,
            -self.m21,
            -self.m22,
            -self.m23,
            -self.m31,
            -self.m32,
            -self.m33
        )

    def __neg__(self):
        return self.neg()

    @cython.nonecheck(False)
    def __richcmp__(v1, v2, int op):
        if op == Py_EQ:
            return  fabs(v1[0][0] - v2[0][0]) < CMP_TOL and fabs(v1[0][1] - v2[0][1]) < CMP_TOL and fabs(v1[0][2] - v2[0][2]) < CMP_TOL  \
                and fabs(v1[1][0] - v2[1][0]) < CMP_TOL and fabs(v1[1][1] - v2[1][1]) < CMP_TOL and fabs(v1[1][2] - v2[1][2]) < CMP_TOL \
                and fabs(v1[2][0] - v2[2][0]) < CMP_TOL and fabs(v1[2][1] - v2[2][1]) < CMP_TOL and fabs(v1[2][2] - v2[2][2]) < CMP_TOL
        elif op == Py_NE:
            return fabs(v1[0][0] - v2[0][0]) >= CMP_TOL or fabs(v1[0][1] - v2[0][1]) >= CMP_TOL or fabs(v1[0][2] - v2[0][2]) >= CMP_TOL  \
                or fabs(v1[1][0] - v2[1][0]) >= CMP_TOL or fabs(v1[1][1] - v2[1][1]) >= CMP_TOL or fabs(v1[1][2] - v2[1][2]) >= CMP_TOL \
                or fabs(v1[2][0] - v2[2][0]) >= CMP_TOL or fabs(v1[2][1] - v2[2][1]) >= CMP_TOL or fabs(v1[2][2] - v2[2][2]) >= CMP_TOL
        raise NotImplementedError("Такой тип сравнения не поддерживается")

    def __getitem__(self, key):
        if key == 0:
            return self.x_axis()
        elif key == 1:
            return self.y_axis()
        elif key == 2:
            return self.z_axis()
        if isinstance(key, str):
            if key == 'm11':
                return self.m11
            if key == 'm12':
                return self.m12
            if key == 'm13':
                return self.m13
            if key == 'm21':
                return self.m21
            if key == 'm22':
                return self.m22
            if key == 'm23':
                return self.m23
            if key == 'm31':
                return self.m31
            if key == 'm32':
                return self.m32
            if key == 'm33':
                return self.m33
        elif isinstance(key, tuple) and (len(key)==2):
            if key[0] == 0 and key[1] == 0:
                return self.m11
            if key[0] == 0 and key[1] == 1:
                return self.m12
            if key[0] == 0 and key[1] == 2:
                return self.m13
            if key[0] == 1 and key[1] == 0:
                return self.m21
            if key[0] == 1 and key[1] == 1:
                return self.m22
            if key[0] == 1 and key[1] == 2:
                return self.m23
            if key[0] == 2 and key[1] == 0:
                return self.m31
            if key[0] == 2 and key[1] == 1:
                return self.m32
            if key[0] == 2 and key[1] == 2:
                return self.m33
        raise KeyError(f'Неправильный индекс {key}')

    cpdef list keys(self):
        return ['m11', 'm12', 'm13', 'm21', 'm22', 'm23', 'm31', 'm32', 'm33']

    def __iter__(self):
        return self.get_generator()

    def get_generator(self):
        yield (self.m11, self.m12, self.m13) 
        yield (self.m21, self.m22, self.m23)   
        yield (self.m31, self.m32, self.m33)  

    @cython.nonecheck(False)
    cpdef Mat3 sub_num_(self, real num):
        self.m11 -= num
        self.m12 -= num
        self.m13 -= num
        self.m21 -= num
        self.m22 -= num
        self.m23 -= num
        self.m31 -= num
        self.m32 -= num
        self.m33 -= num
        return self

    @cython.nonecheck(False)
    cpdef Mat3 sub_num(self, real num):
        return Mat3(
            self.m11 - num,
            self.m12 - num, 
            self.m13 - num,
            self.m21 - num,
            self.m22 - num,
            self.m23 - num,
            self.m31 - num,
            self.m32 - num,
            self.m33 - num
        )


    @cython.nonecheck(False)
    cpdef Mat3 sub_mat_(self, Mat3 mat):
        self.m11 -= mat.m11
        self.m12 -= mat.m12 
        self.m13 -= mat.m13 
        self.m21 -= mat.m21
        self.m22 -= mat.m22
        self.m23 -= mat.m23
        self.m31 -= mat.m31
        self.m32 -= mat.m32
        self.m33 -= mat.m33
        return self

    @cython.nonecheck(False)
    cpdef Mat3 sub_mat(self, Mat3 mat):
        return Mat3(
            self.m11 - mat.m11,
            self.m12 - mat.m12, 
            self.m13 - mat.m13,
            self.m21 - mat.m21,
            self.m22 - mat.m22,
            self.m23 - mat.m23,
            self.m31 - mat.m31,
            self.m32 - mat.m32,
            self.m33 - mat.m33
        )   

    def __sub__(left, right):
        if isinstance(left, Mat3):
            if isinstance(right, Mat3):
                return (<Mat3>left).sub_mat(<Mat3>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Mat3>left).sub_num(<real>right)
        elif isinstance(left, int) or isinstance(left, float):
            return ((<Mat3>right).sub_num(<real>left)).neg()
        raise NotImplementedError(f"Вычесть данные сущности нельзя left={left}, right={right}")
    
    def __isub__(self, right):
        if isinstance(right, Mat3):
            return self.sub_mat_(<Mat3>right)
        elif isinstance(right, int) or isinstance(right, float):
            return self.sub_num_(<real>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя self={self}, right={right}")
    
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat3 div_num_(self, real num):
        self.m11 /= num
        self.m12 /= num
        self.m13 /= num
        self.m21 /= num
        self.m22 /= num
        self.m23 /= num
        self.m31 /= num
        self.m32 /= num
        self.m33 /= num
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat3 div_num(self, real num):
        return Mat3(
            self.m11 / num,
            self.m12 / num, 
            self.m13 / num,
            self.m21 / num,
            self.m22 / num,
            self.m23 / num,
            self.m31 / num,
            self.m32 / num,
            self.m33 / num
        )


    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat3 div_mat_(self, Mat3 mat):
        self.m11 /= mat.m11
        self.m12 /= mat.m12 
        self.m13 /= mat.m13 
        self.m21 /= mat.m21
        self.m22 /= mat.m22
        self.m23 /= mat.m23
        self.m31 /= mat.m31
        self.m32 /= mat.m32
        self.m33 /= mat.m33
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Mat3 div_mat(self, Mat3 mat):
        return Mat3(
            self.m11 / mat.m11,
            self.m12 / mat.m12, 
            self.m13 / mat.m13,
            self.m21 / mat.m21,
            self.m22 / mat.m22,
            self.m23 / mat.m23,
            self.m31 / mat.m31,
            self.m32 / mat.m32,
            self.m33 / mat.m33
        )   

    def __truediv__(left, right):
        if isinstance(left, Mat3):
            if isinstance(right, Mat3):
                return (<Mat3>left).div_mat(<Mat3>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Mat3>left).div_num(<real>right)
        raise NotImplementedError(f"Вычесть данные сущности нельзя left={left}, right={right}")
    
    def __itruediv__(self, right):
        if isinstance(right, Mat3):
            return self.div_mat_(<Mat3>right)
        elif isinstance(right, int) or isinstance(right, float):
            return self.div_num_(<real>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя self={self}, right={right}")


cpdef Quat _convert_quat(object candidate):
    if isinstance(candidate, Quat):
        return <Quat>candidate
    return Quat(candidate[0], candidate[1], candidate[2], candidate[3])

@cython.final
cdef class Quat:
    @classmethod
    def from_list(cls, lst, start_ind=0):
        return cls(lst[start_ind], lst[start_ind+1], lst[start_ind+2], lst[start_ind+3])

    @classmethod
    def from_dict(cls, dct):
        return cls(dct['w'], dct['x'], dct['y'], dct['z'])

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle: real, degrees=False):
        cdef real halfAngle = angle / 2
        if degrees:
            halfAngle /= 180/pi
        cdef real sinus = sin(halfAngle)
        axis = axis.norm()
        return cls(cos(halfAngle), axis.x * sinus, axis.y * sinus, axis.z * sinus)
    
    @classmethod
    def from_two_vecs(cls, u: Vec3, v: Vec3):
        cdef real norm_u_norm_v = sqrt(u.len_sqared() * v.len_sqared())
        cdef Vec3 w = u.cross(v)
        cdef Quat q = Quat(norm_u_norm_v + u.dot(v), w.x, w.y, w.z)
        if q.get_modulus_squared() < CMP_TOL:
            if fabs(u.x) > CMP_TOL:
                q = Quat(0,u.y,-u.x,0)
            elif fabs(u.z) > CMP_TOL:
                q = Quat(0,0,-u.z,u.y)
            else:
                q = Quat(0,0,-u.z,u.y)
        q.norm_()
        return q


    def __cinit__(self, w:real, x:real, y:real, z:real):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f'Quat({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f})'

    def __repr__(self):
        return self.__str__()

    @cython.nonecheck(False)
    cpdef Quat copy(self):
        return Quat(self.w, self.x, self.y, self.z)

    @cython.nonecheck(False)
    cpdef Quat clone(self):
        return Quat(self.w, self.x, self.y, self.z)

    def __getitem__(self, key) -> real:
        if key == 0:
            return self.w
        elif key == 1:
            return self.x
        elif key == 2:
            return self.y
        elif key == 3:
            return self.z
        elif key == 'w':
            return self.w
        elif key =='x':
            return self.x
        elif key == 'y':
            return self.y
        elif key == 'z':
            return self.z
        raise IndexError(f'Невозможно получить компонент вектора по индексу {key}')

    
    def __setitem__(self, key, value: real):
        if key == 0:
            self.w = <real>value
        elif key == 1:
            self.x = <real>value
        elif key == 2:
            self.y = <real>value
        elif key == 3:
            self.z = <real>value
        elif key =='w':
            self.w = <real>value
        elif key =='x':
            self.x = <real>value
        elif key == 'y':
            self.y = <real>value
        elif key == 'z':
            self.z = <real>value
        else:
            raise IndexError(f'Невозможно получить компонент вектора по индексу {key}')
    
    cpdef list keys(self):
        return ['w', 'x', 'y', 'z'] 

    def __len__(self):
        return 4

    def __iter__(self):
        return self.get_generator()

    def get_generator(self):
        yield self.w
        yield self.x
        yield self.y
        yield self.z

    def as_np(self):
        return np.array([self.w, self.x, self.y, self.z])
    
    cpdef tuple as_tuple(self):
        return (self.w, self.x, self.y, self.z)
  
    def to_dict(self) -> dict:
        return {k: self[k] for k in self.keys()}

    @cython.nonecheck(False)
    cpdef real get_modulus(self):
        return sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    @cython.nonecheck(False)
    cpdef real len(self):
        return sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    @cython.nonecheck(False)
    cpdef real get_modulus_squared(self):
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    @cython.nonecheck(False)
    cpdef real len_squared(self):
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    @cython.nonecheck(False)
    def __richcmp__(v1, v2, int op):
        if op == Py_EQ:
            return fabs(v1[0] - v2[0]) < CMP_TOL and fabs(v1[1] - v2[1]) < CMP_TOL \
               and fabs(v1[2] - v2[2]) < CMP_TOL and fabs(v1[2] - v2[2]) < CMP_TOL
        elif op == Py_NE:
            return fabs(v1[0] - v2[0]) >= CMP_TOL or fabs(v1[1] - v2[1]) >= CMP_TOL \
                or fabs(v1[2] - v2[2]) >= CMP_TOL or fabs(v1[2] - v2[2]) >= CMP_TOL
        raise NotImplementedError("Такой тип сравнения не поддерживается")
    
    @cython.nonecheck(False)
    cpdef bint is_eq(self, Quat other):
        return fabs(self.x - other.x) < CMP_TOL and fabs(self.y - other.y) < CMP_TOL \
           and fabs(self.z - other.z) < CMP_TOL and fabs(self.w - other.w) < CMP_TOL

    @cython.nonecheck(False)
    cpdef Quat mul_quat_(self, Quat right):
        cdef real w = self.w * right.w - self.x * right.x - self.y * right.y - self.z * right.z
        cdef real x = self.w * right.x + self.x * right.w + self.y * right.z - self.z * right.y
        cdef real y = self.w * right.y + self.y * right.w + self.z * right.x - self.x * right.z
        cdef real z = self.w * right.z + self.z * right.w + self.x * right.y - self.y * right.x 
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        return self

    @cython.nonecheck(False)
    cpdef Quat mul_quat(self, Quat right):
        return Quat(
            self.w * right.w - self.x * right.x - self.y * right.y - self.z * right.z,
            self.w * right.x + self.x * right.w + self.y * right.z - self.z * right.y,
            self.w * right.y + self.y * right.w + self.z * right.x - self.x * right.z,
            self.w * right.z + self.z * right.w + self.x * right.y - self.y * right.x   
        )

    @cython.nonecheck(False)
    cpdef Vec3 mul_vec(self, Vec3 vec):
        cdef: 
            real num = self.x * 2
            real num2 = self.y * 2
            real num3 = self.z * 2
            real num4 = self.x * num
            real num5 = self.y * num2
            real num6 = self.z * num3
            real num7 = self.x * num2
            real num8 = self.x * num3
            real num9 = self.y * num3
            real num10 = self.w * num
            real num11 = self.w * num2
            real num12 = self.w * num3

        return Vec3(
            (1 - (num5 + num6)) * vec.x + (num7 - num12) * vec.y + (num8 + num11) * vec.z,
            (num7 + num12) * vec.x + (1 - (num4 + num6)) * vec.y + (num9 - num10) * vec.z,
            (num8 - num11) * vec.x + (num9 + num10) * vec.y + (1 - (num4 + num5)) * vec.z
        )

    @cython.nonecheck(False)
    cpdef Quat mul_num_(self, real num):
        self.w *= num
        self.x *= num
        self.y *= num
        self.z *= num
        return self

    @cython.nonecheck(False)
    cpdef Quat mul_num(self, real num):
        cdef Quat res = self.copy()
        return res.mul_num_(num)

    def mul_arr(self, arr):
        cdef int arr_len = len(arr)
        if arr_len == 3:
            return self.mul_vec(_convert(arr))
        if arr_len == 4:
            return self.mul_quat(_convert_quat(arr))
        raise ValueError(f'Нельзя перемножить {self} и {arr}')

    def __mul__(left, right):
        if isinstance(left, Quat):
            if isinstance(right, Quat):
                return (<Quat>left).mul_quat(<Quat>right)
            elif isinstance(right, Vec3):
                return (<Quat>left).mul_vec(<Vec3>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Quat>left).mul_num(<real>right)
            else:
                return (<Quat>left).mul_arr(right)

        elif isinstance(right, Quat):
            if isinstance(left, int) or isinstance(left, float):
                return (<Quat>right).mul_num(<real>left)   
            else:
                return (_convert_quat(left)).mul_quat(<Quat>right)
        raise NotImplementedError(f"Перемножить данные сущности нельзя left={left}, right={right}")
    
    def __imul__(self, other):
        if isinstance(other, Quat):
            return self.mul_quat_(<Quat>other)
        elif isinstance(other, int) or isinstance(other, float):
            return self.mul_num_(<real>other)
        else:
            return self.mul_quat_(_convert_quat(other))

    @cython.nonecheck(False)
    cpdef Vec3 get_axis(self, bint normalize=True):
        cdef Vec3 res = Vec3(self.x, self.y, self.z)
        return res.norm_()

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef real get_angle(self, bint degrees=False):
        cdef real angle = acos(self.w) * 2
        if degrees:
            angle *= 180 / pi
        return angle

    @cython.nonecheck(False)
    cpdef void set_axis(self, Vec3 axis):
        cdef Vec3 norm_axis = axis.norm()
        cdef real len_my_axis = Vec3(self.x, self.y, self.z).len()
        # angle != 0
        if len_my_axis < CMP_TOL:
            return 
        norm_axis.mul_num_(len_my_axis)
        self.x = norm_axis.x
        self.y = norm_axis.y
        self.z = norm_axis.z

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef void set_angle(self, real angle, bint degrees=False):
        cdef Vec3 axis = self.get_axis()
        cdef real halfAngle = angle / 2
        if degrees:
            halfAngle /= 180/pi
        cdef real sinus = sin(halfAngle)
        self.w = cos(halfAngle)
        self.x = axis.x * sinus
        self.y = axis.y * sinus
        self.z = axis.z * sinus

    @property 
    def axis(self):
        return self.get_axis()

    @axis.setter
    def axis(self, value):
        self.set_axis(value)

    @property
    def angle(self):
        return self.get_angle()

    @angle.setter
    def angle(self, value):
        self.set_angle(value)

    @cython.nonecheck(False)
    cpdef Quat conjugate(self):
        return Quat(self.w, -self.x, -self.y, -self.z)

    @cython.nonecheck(False)
    cpdef Quat conjugate_(self):
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Quat inverse_(self):
        cdef real norm = self.get_modulus_squared()
        if norm > CMP_TOL:
            norm = 1/norm
            self.w *= norm
            self.x *= -norm
            self.y *= -norm
            self.z *= -norm
            return self
        raise ValueError(f'Невозможно получить обратный кватернион {self}')
    
    @cython.nonecheck(False)
    cpdef Quat inverse(self):
        cdef Quat res = self.copy()
        return res.inverse_()

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Quat norm_(self, bint raise_zero_len_error=False):
        cdef real norm = self.get_modulus()
        if norm > CMP_TOL:
            norm = 1/norm
            self.w *= norm
            self.x *= -norm
            self.y *= -norm
            self.z *= -norm
            return self
        elif not raise_zero_len_error:
            self.w = 1
            self.x = 0
            self.y = 0
            self.z = 0
            return self
        raise ValueError(f'Невозможно отнормировать кватернион {self}')
        
    @cython.nonecheck(False)
    cpdef Quat norm(self, bint raise_zero_len_error=False):
        cdef Quat res = self.copy()
        return res.norm_(raise_zero_len_error)


    @cython.nonecheck(False)
    cpdef Quat add_num_(self, real num):
        self.w += num
        self.x += num
        self.y += num
        self.z += num
        return self

    @cython.nonecheck(False)
    cpdef Quat add_num(self, real num):
        cdef Quat res = self.copy()
        return res.add_num_(num)

    @cython.nonecheck(False)
    cpdef Quat add_quat_(self, Quat q):
        self.w += q.w
        self.x += q.x
        self.y += q.y
        self.z += q.z
        return self

    @cython.nonecheck(False)
    cpdef Quat add_quat(self, Quat q):
        cdef Quat res = self.copy()
        return res.add_quat_(q)   

    def add_arr_(self, arr):
        cdef int arr_len = len(arr)
        if arr_len == 4:
            return self.add_quat_(_convert_quat(arr))
        raise ValueError(f'Нельзя сложить {self} и {arr}')
    
    def add_arr(self, arr):
        cdef int arr_len = len(arr)
        if arr_len == 4:
            return self.add_quat(_convert_quat(arr))
        raise ValueError(f'Нельзя сложить {self} и {arr}')

    def __add__(left, right):
        if isinstance(left, Quat):
            if isinstance(right, Quat):
                return (<Quat>left).add_quat(<Quat>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Quat>left).add_num(<real>right)
            else:
                return (<Quat>left).add_arr(right)

        elif isinstance(right, Quat):
            if isinstance(left, int) or isinstance(left, float):
                return (<Quat>right).add_num(<real>left)   
            else:
                return (_convert_quat(left)).add_quat(<Quat>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя left={left}, right={right}")
    
    def __iadd__(self, other):
        if isinstance(other, Quat):
            return self.add_quat_(<Quat>other)
        elif isinstance(other, int) or isinstance(other, float):
            return self.add_num_(<real>other)
        else:
            return self.add_quat_(_convert_quat(other))
    
    @cython.nonecheck(False)
    cpdef Quat sub_num_(self, real num):
        self.w -= num
        self.x -= num
        self.y -= num
        self.z -= num
        return self

    @cython.nonecheck(False)
    cpdef Quat sub_num(self, real num):
        cdef Quat res = self.copy()
        return res.sub_num_(num)

    @cython.nonecheck(False)
    cpdef Quat sub_quat_(self, Quat q):
        self.w -= q.w
        self.x -= q.x
        self.y -= q.y
        self.z -= q.z
        return self

    @cython.nonecheck(False)
    cpdef Quat sub_quat(self, Quat q):
        cdef Quat res = self.copy()
        return res.sub_quat_(q)   

    def sub_arr_(self, arr):
        cdef int arr_len = len(arr)
        if arr_len == 4:
            return self.sub_quat_(_convert_quat(arr))
        raise ValueError(f'Нельзя вычесть {self} и {arr}')
    
    def sub_arr(self, arr):
        cdef int arr_len = len(arr)
        if arr_len == 4:
            return self.sub_quat(_convert_quat(arr))
        raise ValueError(f'Нельзя вычесть {self} и {arr}')

    def __sub__(left, right):
        if isinstance(left, Quat):
            if isinstance(right, Quat):
                return (<Quat>left).sub_quat(<Quat>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Quat>left).sub_num(<real>right)
            else:
                return (<Quat>left).sub_arr(right)

        elif isinstance(right, Quat):
            if isinstance(left, int) or isinstance(left, float):
                return (Quat(left, left, left, left)).sub_quat(right)   
            else:
                return (_convert_quat(left)).sub_quat(<Quat>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя left={left}, right={right}")
    
    def __isub__(self, other):
        if isinstance(other, Quat):
            return self.sub_quat_(<Quat>other)
        elif isinstance(other, int) or isinstance(other, float):
            return self.sub_num_(<real>other)
        else:
            return self.sub_quat_(_convert_quat(other))

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Quat div_num_(self, real num):
        self.w /= num
        self.x /= num
        self.y /= num
        self.z /= num
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Quat div_num(self, real num):
        cdef Quat res = self.copy()
        return res.div_num_(num)

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Quat div_quat_(self, Quat q):
        self.w /= q.w
        self.x /= q.x
        self.y /= q.y
        self.z /= q.z
        return self

    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef Quat div_quat(self, Quat q):
        cdef Quat res = self.copy()
        return res.div_quat_(q)   

    def div_arr_(self, arr):
        cdef int arr_len = len(arr)
        if arr_len == 4:
            return self.div_quat_(_convert_quat(arr))
        raise ValueError(f'Нельзя вычесть {self} и {arr}')
    
    def div_arr(self, arr):
        cdef int arr_len = len(arr)
        if arr_len == 4:
            return self.div_quat(_convert_quat(arr))
        raise ValueError(f'Нельзя вычесть {self} и {arr}')

    def __truediv__(left, right):
        if isinstance(left, Quat):
            if isinstance(right, Quat):
                return (<Quat>left).div_quat(<Quat>right)
            elif isinstance(right, int) or isinstance(right, float):
                return (<Quat>left).div_num(<real>right)
            else:
                return (<Quat>left).div_arr(right)

        elif isinstance(right, Quat):
            if isinstance(left, int) or isinstance(left, float):
                return (Quat(left, left, left, left)).div_quat(right)   
            else:
                return (_convert_quat(left)).div_quat(<Quat>right)
        raise NotImplementedError(f"Сложить данные сущности нельзя left={left}, right={right}")
    
    def __itruediv__(self, other):
        if isinstance(other, Quat):
            return self.div_quat_(<Quat>other)
        elif isinstance(other, int) or isinstance(other, float):
            return self.div_num_(<real>other)
        else:
            return self.div_quat_(_convert_quat(other))

    @cython.nonecheck(False)
    cpdef Quat neg_(self):
        return self.mul_num_(-1)
    
    @cython.nonecheck(False)
    cpdef Quat neg(self):
        return self.mul_num(-1)    

    def __neg__(self) -> Quat:
        return self.neg()

    @cython.nonecheck(False)
    cpdef real dot(self, Quat q):
        return self.w * q.w + self.x * q.x + self.y * q.y +self.z * q.z

    @cython.nonecheck(False)
    cpdef Mat3 to_Mat3(self):
        cdef:
            real x = self.x
            real y = self.y
            real z = self.z
            real w = self.w
            real xx = x * x
            real yy = y * y
            real zz = z * z
            real xy = x * y
            real xz = x * z
            real yz = y * z
            real wx = w * x
            real wy = w * y
            real wz = w * z
        return Mat3(
            1 - 2 * yy - 2 * zz,2 * xy - 2 * wz,2 * xz + 2 * wy,
            2 * xy + 2 * wz,1 - 2 * xx - 2 * zz,2 * yz - 2 * wx,
            2 * xz - 2 * wy,2 * yz + 2 * wx,1 - 2 * xx - 2 * yy)




    



    
