from .vectors cimport Vec2, real, rational

cpdef bint is_bbox_intersect(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 u2)
cpdef bint is_cross_lines(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 u2)
cpdef bint is_point_on_the_line(Vec2 u1, Vec2 u2, Vec2 p)

cpdef real fmax(real a, real b)
cpdef real fmin(real a, real b)

cdef class Rect:
    cdef public real x1, x2, y1, y2
    cpdef Rect copy(self)
    cpdef Rect clone(self)
    cpdef real[:] as_np(self)
    cpdef tuple as_tuple(self)
    cpdef list keys(self)
    cpdef bint is_null(self)
    cpdef bint is_in(self, Vec2 p)
    cpdef bint is_cross_seg(self, Vec2 p1, Vec2 p2)
    cpdef bint is_cross_ray(self, Vec2 p1, Vec2 p2)
    cpdef bint is_cross_line(self, Vec2 p1, Vec2 p2)
    cpdef real area(self)
    cpdef real perimeter(self)
    cpdef bint is_intersect(self, Rect other)
    cpdef Rect intersect(self, Rect other)
    cpdef Rect union(self, Rect other)
    cpdef Rect union_vec(self, Vec2 vec)
    cpdef Rect union_point(self, real x, real y)