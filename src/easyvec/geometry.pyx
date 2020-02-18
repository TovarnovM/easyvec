import numpy as np
from libc.math cimport fabs
cimport cython
from .vectors cimport CMP_TOL


@cython.nonecheck(False)
cpdef bint is_bbox_intersect(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2):
    return (fmin(u1.x, u2.x) <= fmax(v1.x, v2.x)) and (fmax(u1.x, u2.x) >= fmin(v1.x, v2.x)) \
       and (fmin(u1.y, u2.y) <= fmax(v1.y, v2.y)) and (fmax(u1.y, u2.y) >= fmin(v1.y, v2.y))    

@cython.nonecheck(False)
@cython.cdivision(True)
cpdef (bint, real, real) _intersect_ts(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2):
    cdef Vec2 vec1 = u1.sub(v1)
    cdef Vec2 vec2 = v2.sub(v1)
    cdef Vec2 vec3 = u2.sub(u1).rotate90()

    cdef real dot = vec2.dot(vec3)
    if fabs(dot) < CMP_TOL:
        if vec1.len_sqared() <= CMP_TOL:
            return True, 0.0, 0.0
        return False, 0.0, 0.0
    
    cdef real t1 = vec2.cross(vec1) / dot
    cdef real t2 = vec1.dot(vec3) / dot
    return True, t1, t2

@cython.nonecheck(False)
cpdef Vec2 intersect_lines(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(u1, u2, v1, v2)
    if not suc:
        return None
    return u1.add( u2.sub(u1).mul(t1) )

@cython.nonecheck(False)
cpdef Vec2 intersect_segments(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2):
    if not is_bbox_intersect(u1, u2, v1, v2):
        return None
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(u1, u2, v1, v2)
    if (not suc) or t1 < 0.0 or t1 > 1.0 or t2 < 0.0 or t2 > 1.0 :
        return None
    return u1.add( u2.sub(u1).mul(t1) )

@cython.nonecheck(False)
cpdef Vec2 intersect_rays(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(u1, u2, v1, v2)
    if (not suc) or t1 < 0.0 or t2 < 0.0 :
        return None
    return u1.add( u2.sub(u1).mul(t1) )

@cython.nonecheck(False)
cpdef Vec2 intersect_ray_line(Vec2 r1, Vec2 r2, Vec2 v1, Vec2 v2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(r1, r2, v1, v2)
    if (not suc) or t1 < 0.0:
        return None
    return r1.add( r2.sub(r1).mul(t1) )    

@cython.nonecheck(False)
cpdef Vec2 intersect_ray_segment(Vec2 r1, Vec2 r2, Vec2 v1, Vec2 v2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(r1, r2, v1, v2)
    if (not suc) or t1 < 0.0 or t2 < 0.0 or t2 > 1.0 :
        return None
    return r1.add( r2.sub(r1).mul(t1) )   

@cython.nonecheck(False)
cpdef Vec2 intersect_line_segment(Vec2 u1, Vec2 u2, Vec2 s1, Vec2 s2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(u1, u2, s1, s2)
    if (not suc) or t2 < 0.0 or t2 > 1.0 :
        return None
    return u1.add( u2.sub(u1).mul(t1) )

cpdef real fmax(real a, real b):
    if a > b:
        return a
    else:
        return b

cpdef real fmin(real a, real b):
    if a < b:
        return a
    else:
        return b


cdef class Rect:
    def __cinit__(self, *args):
        cdef int alen = len(args)
        if alen >= 4:
            self.x1, self.y1 = <real>(args[0]), <real>(args[1])
            self.x2, self.y2 = <real>(args[2]), <real>(args[3])
        elif alen == 2:
            self.x1, self.x2 = <real>(args[0][0]), <real>(args[1][0])
            self.y1, self.y2 = <real>(args[0][1]), <real>(args[1][1])
        elif alen == 1:
            self.x1, self.y1 = <real>(args[0][0]), <real>(args[0][1])
            self.x2, self.y2 = <real>(args[0][2]), <real>(args[0][3])
        else:
            raise ValueError(f'Невозможно создать экземпляр Rect из параметров {args}')
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    def __str__(self):
        return f'Rect({self.x1:.2f}, {self.y1:.2f}, {self.x2:.2f}, {self.y2:.2f})'

    def __repr__(self):
        return str(self)

    cpdef Rect copy(self):
        return Rect(self.x1, self.y1, self.x2, self.y2)

    cpdef Rect clone(self):
        return Rect(self.x1, self.y1, self.x2, self.y2)

    cpdef real[:] as_np(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    cpdef tuple as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def __getitem__(self, key) -> real:
        if key == 0:
            return self.x1
        elif key == 1:
            return self.y1
        elif key == 2:
            return self.x2
        elif key == 3:
            return self.y2
        elif key =='x1':
            return self.x1
        elif key == 'y1':
            return self.y1
        elif key == 'x2':
            return self.x2
        elif key == 'y2':
            return self.y2
        raise IndexError(f'Невозможно получить компонент прямоугольника по индексу {key}')

    def __setitem__(self, key, value: real):
        if key == 0:
            self.x1 = <real>value
        elif key == 1:
            self.y1 = <real>value
        elif key == 2:
            self.x2 = <real>value
        elif key == 3:
            self.y2 = <real>value
        elif key =='x1':
            self.x1 = <real>value
        elif key == 'x2':
            self.x2 = <real>value
        elif key == 'y1':
            self.y1 = <real>value
        elif key == 'y2':
            self.y2 = <real>value
        else:
            raise IndexError(f'Невозможно получить компонент прямоугольника по индексу {key}')
    
    cpdef list keys(self):
        return ['x1', 'y1', 'x2', 'y2'] 

    def __iter__(self):
        return self.get_generator()

    def get_generator(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    cpdef bint is_null(self):
        return fabs(self.x1) < CMP_TOL and fabs(self.y1) < CMP_TOL and \
               fabs(self.x2) < CMP_TOL and fabs(self.y2) < CMP_TOL

    cpdef bint is_in(self, Vec2 p):
        return (self.x1 <= p.x <= self.x2) and (self.y1 <= p.y <= self.y2)

    cpdef bint is_cross_seg(self, Vec2 p1, Vec2 p2):
        pass

    cpdef bint is_cross_ray(self, Vec2 p1, Vec2 p2):
        pass
    
    cpdef bint is_cross_line(self, Vec2 p1, Vec2 p2):
        if p1.is_eq(p2):
            return 0

    cpdef real area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    cpdef real perimeter(self):
        return 2*(self.x2 - self.x1 + self.y2 - self.y1)
    
    @cython.nonecheck(False)
    cpdef bint is_intersect(self, Rect other):
        return (self.x1 <= other.x2) and (self.x2 >= other.x1) \
           and (self.y1 <= other.y2) and (self.y2 >= other.y2)

    @cython.nonecheck(False)
    cpdef Rect intersect(self, Rect other):
        if self.is_null() or other.is_null() or (not self.is_intersect(other)):
            return Rect(0.0, 0.0, 0.0, 0.0)
        cdef real nx1 = fmax(self.x1, other.x1)
        cdef real ny1 = fmax(self.y1, other.y1)
        cdef real nx2 = fmin(self.x2, other.x2)
        cdef real ny2 = fmin(self.y2, other.y2)
        return Rect(nx1, ny1, nx2, ny2)

    def __mul__(r1, r2):
        if isinstance(r1, Rect) and isinstance(r2, Rect):
            return (<Rect>r1).intersect(<Rect>r2)
        else:
            raise ValueError(f'Невозможно пересечь сущности {r1} и {r2}')

    @cython.nonecheck(False)
    cpdef Rect union(self, Rect other):
        if other.is_null():
            return self.copy()
        if self.is_null():
            return other.copy()

        cdef real nx1 = fmin(self.x1, other.x1)
        cdef real ny1 = fmin(self.y1, other.y1)
        cdef real nx2 = fmax(self.x2, other.x2)
        cdef real ny2 = fmax(self.y2, other.y2)
        return Rect(nx1, ny1, nx2, ny2)

    @cython.nonecheck(False)
    cpdef Rect union_vec(self, Vec2 vec):
        return self.union_point(vec.x, vec.y)

    @cython.nonecheck(False)
    cpdef Rect union_point(self, real x, real y):
        if self.is_null():
            return Rect(x, y, x, y)
        cdef real nx1 = fmin(self.x1, x)
        cdef real ny1 = fmin(self.y1, y)
        cdef real nx2 = fmax(self.x2, x)
        cdef real ny2 = fmax(self.y2, y)
        return Rect(nx1, ny1, nx2, ny2)

    def __sum__(r1, r2):
        if isinstance(r1, Rect):
            if isinstance(r2, Rect):
                return (<Rect>r1).union(<Rect>r2)
            if isinstance(r2, Vec2):
                return (<Rect>r1).union_vec(<Vec2>r2)
            elif isinstance(r2, np.ndarray) or isinstance(r2, tuple) or isinstance(r2, list) or isinstance(r2, memoryview):
                return (<Rect>r1).union_point(<real>(r2[0]), <real>(r2[1]))
        raise ValueError(f'Невозможно объединить сущности {r1} и {r2}')
            