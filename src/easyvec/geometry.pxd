from .vectors cimport Vec2, real, rational
from cpython cimport array
import array

cpdef Vec2 _convert(object candidate)
cpdef bint is_bbox_intersect(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2)
cpdef (bint, real, real) _intersect_ts(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2)
cpdef Vec2 intersect_lines(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2)
cpdef Vec2 intersect_segments(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2)
cpdef Vec2 intersect_rays(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2)
cpdef Vec2 intersect_ray_line(Vec2 r1, Vec2 r2, Vec2 v1, Vec2 v2)
cpdef Vec2 intersect_ray_segment(Vec2 r1, Vec2 r2, Vec2 v1, Vec2 v2)
cpdef Vec2 intersect_line_segment(Vec2 u1, Vec2 u2, Vec2 s1, Vec2 s2)

cpdef real fmax(real a, real b) nogil
cpdef real fmin(real a, real b) nogil
cpdef array.array _sortreduce(real[:] dists)
cpdef list _sortreduce_by_distance(list vecs, Vec2 close_to_point)

cdef class Rect:
    cdef public real x1, x2, y1, y2
    cpdef Rect copy(self)
    cpdef Rect clone(self)
    cpdef tuple as_tuple(self)
    cpdef list keys(self)
    cpdef bint is_null(self)
    cpdef bint is_in(self, Vec2 p)
    cpdef Vec2 intersect_general(self, Vec2 p1, Vec2 p2, real f_low, real f_high,bint ret_closest=*)
    cpdef Vec2 intersect_segment(self, Vec2 p1, Vec2 p2, bint ret_closest=*)
    cpdef Vec2 intersect_ray(self, Vec2 p1, Vec2 p2, bint ret_closest=*)
    cpdef Vec2 intersect_line(self, Vec2 p1, Vec2 p2, bint ret_closest=*)
    cpdef dict to_dict(self)
    cpdef real area(self)
    cpdef real perimeter(self)
    cpdef bint is_intersect_rect(self, Rect other)
    cpdef Rect intersect_rect(self, Rect other)
    cpdef Rect union(self, Rect other)
    cpdef Rect union_vec(self, Vec2 vec)
    cpdef Rect union_point(self, real x, real y)
    cpdef bint is_bbox_intersect(self, Vec2 p1, Vec2 p2)

cdef class PolyLine:
    cdef public list vecs
    cdef public bint enclosed
    cdef Rect bbox
    cdef int vlen
    cpdef list intersect_general(self, Vec2 p1, Vec2 p2, real f_low, real f_high, bint sortreduce=*)
    cpdef list intersect_line(self, Vec2 p1, Vec2 p2, bint sortreduce=*)
    cpdef list intersect_ray(self, Vec2 p1, Vec2 p2, bint sortreduce=*)
    cpdef list intersect_segment(self, Vec2 p1, Vec2 p2, bint sortreduce=*)