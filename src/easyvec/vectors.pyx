
from libc.math cimport fabs
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_GE, Py_GT, Py_NE


cdef real CMP_TOL = 1e-6
def get_CMP_TOL():
    return CMP_TOL

cdef class Vec2:
    def __cinit__(self, x: real, y:real):
        self.x = x
        self.y = y

    def __str__(self):
        return f'({self.x:.2f}, {self.y:.2f})'

    def __repr__(self):
        return f'Vec2({self.x}, {self.y})'

    def __richcmp__(Vec2 v1, Vec2 v2, int op):
        if op == Py_EQ:
            return fabs(v1.x - v2.x) < CMP_TOL and fabs(v1.y - v2.y) < CMP_TOL 
        elif op == Py_NE:
            return fabs(v1.x - v2.x) >= CMP_TOL or fabs(v1.y - v2.y) >= CMP_TOL
        raise NotImplementedError("Такой тип сравнения не поддерживается")


cdef class Vec3:
    def __cinit__(self, x: real, y: real, z: real):
        self.x = x
        self.y = y
        self.z = z 

    def __str__(self):
        return f'({self.x:.2f}, {self.y:.2f}, {self.z:.2f})'

    def __repr__(self):
        return f'Vec3({self.x}, {self.y}, {self.z})'