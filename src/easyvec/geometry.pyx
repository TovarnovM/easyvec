import numpy as np
from libc.math cimport fabs
cimport cython
from .vectors cimport CMP_TOL, real, Vec2, rational, BIG_REAL, MINUS_BIG_REAL
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_GE, Py_GT, Py_NE

cdef array.array _int_array_template = array.array('i', [])
cdef array.array _double_array_template = array.array('d', [])

cpdef Vec2 _convert(object candidate):
    if isinstance(candidate, Vec2):
        return <Vec2>(candidate)
    else:
        return Vec2(candidate[0], candidate[1])

def intersect(*args, **kwargs):
    cdef Vec2 u1, u2, v1, v2
    cdef int alen = len(args)
    cdef int klen = len(kwargs)
    cdef str key1, key2
    cdef object some1, some2
    cdef set line_set = {'line', 'l', 'line1', 'l1', 'line2', 'l2'}
    cdef set segment_set = {'segment', 's', 'segment1', 's1', 'segment2', 's2'}
    cdef set ray_set = {'ray', 'r', 'ray1', 'r1', 'ray2', 'r2'}
    
    if alen == 4 and klen == 0:
        u1 = _convert(args[0])
        u2 = _convert(args[1])
        v1 = _convert(args[2])
        v2 = _convert(args[3])
        return intersect_segments(u1, u2, v1, v2)
    if alen == 2 and klen == 1:
        u1 = _convert(args[0])
        u2 = _convert(args[1])

        for k in kwargs:
            v1 = _convert(kwargs[k][0])
            v2 = _convert(kwargs[k][1])
            if k in line_set:
                return intersect_line_segment(v1, v2, u1, u2)
            elif k in segment_set:
                return intersect_segments(v1, v2, u1, u2)  
            elif k in ray_set:
                return intersect_ray_segment(v1, v2, u1, u2)            

        raise ValueError(f'Неправильные аргументы {args} {kwargs}')
         
    elif alen == 0 and klen == 2:
        key1, (some1, some2) = kwargs.popitem()
        u1 = _convert(some1)
        u2 = _convert(some2)
        key2, (some1, some2) = kwargs.popitem()
        v1 = _convert(some1)
        v2 = _convert(some2)
        if key1 in line_set:
            if key2 in segment_set:
                return intersect_line_segment(u1, u2, v1, v2)
            elif key2 in line_set:
                return intersect_lines(u1, u2, v1, v2)
            elif key2 in ray_set:
                return intersect_ray_line(v1, v2, u1, u2)
        elif key1 in segment_set:
            if key2 in segment_set:
                return intersect_segments(u1, u2, v1, v2)
            elif key2 in line_set:
                return intersect_line_segment(v1, v2, u1, u2)
            elif key2 in ray_set:
                return intersect_ray_segment(v1, v2, u1, u2) 
        elif key1 in ray_set:
            if key2 in segment_set:
                return intersect_ray_segment(u1, u2, v1, v2)
            elif key2 in line_set:
                return intersect_ray_line(u1, u2, v1, v2)
            elif key2 in ray_set:
                return intersect_rays(v1, v2, u1, u2)
    raise ValueError(f'Неправильные аргументы {args} {kwargs}')  


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
    return u1.add( u2.sub(u1).mul_num(t1) )

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
    return u1.add( u2.sub(u1).mul_num(t1) )

@cython.nonecheck(False)
cpdef Vec2 intersect_rays(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(u1, u2, v1, v2)
    if (not suc) or t1 < 0.0 or t2 < 0.0 :
        return None
    return u1.add( u2.sub(u1).mul_num(t1) )

@cython.nonecheck(False)
cpdef Vec2 intersect_ray_line(Vec2 r1, Vec2 r2, Vec2 v1, Vec2 v2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(r1, r2, v1, v2)
    if (not suc) or t1 < 0.0:
        return None
    return r1.add( r2.sub(r1).mul_num(t1) )    

@cython.nonecheck(False)
cpdef Vec2 intersect_ray_segment(Vec2 r1, Vec2 r2, Vec2 v1, Vec2 v2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(r1, r2, v1, v2)
    if (not suc) or t1 < 0.0 or t2 < 0.0 or t2 > 1.0 :
        return None
    return r1.add( r2.sub(r1).mul_num(t1) )   

@cython.nonecheck(False)
cpdef Vec2 intersect_line_segment(Vec2 u1, Vec2 u2, Vec2 s1, Vec2 s2):
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(u1, u2, s1, s2)
    if (not suc) or t2 < 0.0 or t2 > 1.0 :
        return None
    return u1.add( u2.sub(u1).mul_num(t1) )

cpdef inline real fmax(real a, real b) nogil:
    if a > b:
        return a
    else:
        return b

cpdef inline real fmin(real a, real b) nogil:
    if a < b:
        return a
    else:
        return b

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef array.array _sortreduce(real[:] dists): 
    cdef int lst_len = dists.shape[0]
    if lst_len <= 1:
        return array.clone(_int_array_template, 1, zero=True)
    cdef array.array res = array.clone(_int_array_template, lst_len, zero=False)
    cdef array.array res2 = array.clone(_int_array_template, lst_len, zero=False)
    cdef int[:] ind_arr = res
    cdef int[:] ind_arr2 = res2
    cdef int i
    for i in range(lst_len):
        ind_arr[i] = i
    cdef real bufferr
    cdef int j, bufferi, i_min
    for j in range(lst_len - 1):
        i_min = j
        for i in range(j + 1, lst_len):
            if dists[i] < dists[i_min]:
                i_min = i
        ind_arr[j], ind_arr[i_min] = ind_arr[i_min], ind_arr[j]
        dists[j], dists[i_min] = dists[i_min], dists[j]
    ind_arr2[0] = ind_arr[0]
    j = 0
    for i in range(1, lst_len):
        if fabs(dists[i] - dists[i-1]) > CMP_TOL:
            j = j + 1
            ind_arr2[j] = ind_arr[i]
    if j+1 != lst_len:
        array.resize(res2, j+1)
    return res2       

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list _sortreduce_by_distance(list vecs, Vec2 close_to_point):
    cdef int vec_len = len(vecs)
    if vec_len < 2:
        return vecs
    cdef list res2 = []
    cdef array.array distances, inds
    cdef real[:] dists

    distances = array.clone(_double_array_template, vec_len, zero=False)
    for i in range(vec_len):
        distances[i] = (close_to_point.sub(<Vec2>(vecs[i]))).len_sqared()
    dists = distances
    inds = _sortreduce(dists)
    for i in range(len(inds)):
        res2.append(vecs[inds[i]])
    return res2    


@cython.final
cdef class Rect:
    @classmethod
    def null(cls) -> Rect:
        return cls(0,0,0,0)

    @classmethod
    def from_dict(cls, dct: dict) -> Rect:
        return cls(dct['x1'], dct['y1'], dct['x2'], dct['y2'])

    @classmethod
    def bbox(cls, *args):
        cdef int alen = len(args)
        cdef int alen2, i 
        cdef real x1, y1, x2, y2
        cdef Vec2 v
        if alen == 1:
            alen2 = len(args[0])
            if alen2 < 2:
                return cls.null()
            v = _convert(args[0][0])
            x1, y1, x2, y2 = v.x, v.y, v.x, v.y
            for i in range(1, alen2):
                v = _convert(args[0][i])
                x1 = fmin(x1, v.x)
                y1 = fmin(y1, v.y)
                x2 = fmax(x2, v.x)
                y2 = fmax(y2, v.y)
            return cls(x1, y1, x2, y2)
        elif alen > 1:
            v = _convert(args[0])
            x1, y1, x2, y2 = v.x, v.y, v.x, v.y
            for i in range(1, alen):
                v = _convert(args[i])
                x1 = fmin(x1, v.x)
                y1 = fmin(y1, v.y)
                x2 = fmax(x2, v.x)
                y2 = fmax(y2, v.y)
            return cls(x1, y1, x2, y2)
        else:
            return cls.null()

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

    def as_np(self):
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

    cpdef dict to_dict(self):
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
        }

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

    @cython.nonecheck(False)
    cpdef bint is_in(self, Vec2 p):
        return (self.x1 <= p.x <= self.x2) and (self.y1 <= p.y <= self.y2)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef Vec2 intersect_general(self, Vec2 p1, Vec2 p2, real f_low, real f_high,bint ret_closest=True):
        cdef real znam = p2.x - p1.x
        cdef real f_dim_low, f_dim_high

        # segm not parallel Ox
        if fabs(znam) > CMP_TOL:
            f_dim_low = (self.x1 - p1.x) / znam
            f_dim_high= (self.x2 - p1.x) / znam     

            if f_dim_high < f_dim_low:
                f_dim_high, f_dim_low = f_dim_low, f_dim_high # Swap
            
            if f_dim_high < f_low or f_dim_low > f_high:
                return None
            f_low = fmax(f_dim_low, f_low)
            f_high= fmin(f_dim_high, f_high)

            if f_low > f_high:
                return None
        
        znam = p2.y - p1.y
        # segm not parallel Oy
        if fabs(znam) > CMP_TOL:
            f_dim_low = (self.y1 - p1.y) / znam
            f_dim_high= (self.y2 - p1.y) / znam     

            if f_dim_high < f_dim_low:
                f_dim_high, f_dim_low = f_dim_low, f_dim_high # Swap
            
            if f_dim_high < f_low or f_dim_low > f_high:
                return None
            f_low = fmax(f_dim_low, f_low)
            f_high= fmin(f_dim_high, f_high)

            if f_low > f_high:
                return None
        
        if ret_closest:
            return p1.add( p2.sub(p1).mul_num(f_low) )
        else:
            return p1.add( p2.sub(p1).mul_num(f_high) )

    @cython.nonecheck(False)
    cpdef Vec2 intersect_segment(self, Vec2 p1, Vec2 p2, bint ret_closest=True):
        return self.intersect_general(p1, p2, 0.0, 1.0, ret_closest)

    @cython.nonecheck(False)
    cpdef Vec2 intersect_ray(self, Vec2 p1, Vec2 p2, bint ret_closest=True):
        return self.intersect_general(p1, p2, 0.0, BIG_REAL, ret_closest)

    @cython.nonecheck(False)
    cpdef Vec2 intersect_line(self, Vec2 p1, Vec2 p2, bint ret_closest=True):
        return self.intersect_general(p1, p2, MINUS_BIG_REAL, BIG_REAL, ret_closest)

    cpdef real area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    cpdef real perimeter(self):
        return 2*(self.x2 - self.x1 + self.y2 - self.y1)
    
    @cython.nonecheck(False)
    cpdef bint is_intersect_rect(self, Rect other):
        return (self.x1 <= other.x2) and (self.x2 >= other.x1) \
           and (self.y1 <= other.y2) and (self.y2 >= other.y2)

    @cython.nonecheck(False)
    cpdef Rect intersect_rect(self, Rect other):
        if self.is_null() or other.is_null() or (not self.is_intersect_rect(other)):
            return Rect(0.0, 0.0, 0.0, 0.0)
        cdef real nx1 = fmax(self.x1, other.x1)
        cdef real ny1 = fmax(self.y1, other.y1)
        cdef real nx2 = fmin(self.x2, other.x2)
        cdef real ny2 = fmin(self.y2, other.y2)
        return Rect(nx1, ny1, nx2, ny2)

    def __mul__(r1, r2):
        if isinstance(r1, Rect):
            return (<Rect>r1).intersect(r2)
        elif isinstance(r2, Rect):
            return (<Rect>r2).intersect(r1)
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

    def __add__(r1, r2):
        if isinstance(r1, Rect):
            if isinstance(r2, Rect):
                return (<Rect>r1).union(<Rect>r2)
            if isinstance(r2, Vec2):
                return (<Rect>r1).union_vec(<Vec2>r2)
            elif isinstance(r2, np.ndarray) or isinstance(r2, tuple) or isinstance(r2, list) or isinstance(r2, memoryview):
                return (<Rect>r1).union_point(<real>(r2[0]), <real>(r2[1]))
        raise ValueError(f'Невозможно объединить сущности {r1} и {r2}')
            
    @cython.nonecheck(False)
    def __richcmp__(r1, r2, int op):
        if op == Py_EQ:
            return fabs((<Rect>r1).x1 - (<Rect>r2).x1) < CMP_TOL \
               and fabs((<Rect>r1).x2 - (<Rect>r2).x2) < CMP_TOL \
               and fabs((<Rect>r1).y1 - (<Rect>r2).y1) < CMP_TOL \
               and fabs((<Rect>r1).y2 - (<Rect>r2).y2) < CMP_TOL 

        elif op == Py_NE:
            return fabs((<Rect>r1).x1 - (<Rect>r2).x1) >= CMP_TOL \
                or fabs((<Rect>r1).x2 - (<Rect>r2).x2) >= CMP_TOL \
                or fabs((<Rect>r1).y1 - (<Rect>r2).y1) >= CMP_TOL \
                or fabs((<Rect>r1).y2 - (<Rect>r2).y2) >= CMP_TOL 
        raise NotImplementedError("Такой тип сравнения не поддерживается")

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def intersect(self, *args, **kwargs):
        cdef int alen = len(args)
        if alen == 1:
            if isinstance(args[0], Rect):
                return self.intersect_rect(args[0])
            v1, v2 = args[0]
            v1 = _convert(v1)
            v2 = _convert(v2)
            return self.intersect_segment(v1, v2)
        elif alen == 2:
            v1, v2 = args
            v1 = _convert(v1)
            v2 = _convert(v2)
            return self.intersect_segment(v1, v2)
        elif alen > 2:
            raise  ValueError(f'Неправильные аргументы {args} {kwargs}')  
        cdef int klen = len(kwargs)
        cdef str key
        if klen == 1:
            key, tp = kwargs.popitem()
            if key in {'rect'}:
                return self.intersect_rect(tp)
            v1, v2 = tp
            v1 = _convert(v1)
            v2 = _convert(v2)
            if key in {'line', 'l' }:
                return self.intersect_line(v1, v2)
            if key in {'ray', 'r' }:
                return self.intersect_ray(v1, v2)
            if key in {'segment', 's' }:
                return self.intersect_segment(v1, v2)
        else:
            raise  ValueError(f'Неправильные аргументы {args} {kwargs}')  
       
    @cython.nonecheck(False)
    cpdef bint is_bbox_intersect(self, Vec2 p1, Vec2 p2):
        return (self.x1 <= fmax(p1.x, p2.x)) and (self.x2 >= fmin(p1.x, p2.x)) \
           and (self.y1 <= fmax(p1.y, p2.y)) and (self.y2 >= fmin(p1.y, p2.y))
 

@cython.final
cdef class PolyLine:
    def __cinit__(self, vecs: list, enclosed=True, copy_data=False):
        cdef int vec_len, i
        if copy_data:
            self.vecs = []
            vec_len = len(vecs)
            for i in range(vec_len):
                self.vecs.append( (<Vec2>(vecs[i])).copy() )
        else:
            self.vecs = vecs
        self.vlen = len(self.vecs)
        if self.vlen < 2:
            raise ValueError(f'Слишком мало точек для линии. Необходимо больше, чем 1')
        self.enclosed = enclosed
        self.bbox = Rect.bbox(self.vecs)

    def __str__(self):
        s = [f'({v.x:.2f}, {v.y:.2f})' for v in self.vecs]
        s = ', '.join(s)
        return f'PolyLine({s})'

    def __repr__(self):
        s = [f'({v.x}, {v.y})' for v in self.vecs]
        s = ', '.join(s)
        return f'PolyLine(vecs=[{s}], enclosed={self.enclosed})'

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list intersect_general(self, Vec2 p1, Vec2 p2, real f_low, real f_high, bint sortreduce=True):
        cdef list res = []
        if self.bbox.intersect_general(p1, p2, f_low, f_high) is None:
            return res
        cdef int i
        cdef Vec2 v1, v2, v_cr
        cdef bint inter
        cdef real t1, t2
        v1 = <Vec2>(self.vecs[0])
        for i in range(1, self.vlen):
            v2 = <Vec2>(self.vecs[i])
            inter, t1, t2 = _intersect_ts(p1, p2, v1, v2)
            if inter and (0.0 <= t2 <= 1.0) and (f_low <= t1 <= f_high):
                v_cr = p1.add( p2.sub(p1).mul_num(t1) )
                res.append(v_cr)
            v1 = v2
        if self.enclosed:
            v2 = <Vec2>(self.vecs[0])
            inter, t1, t2 = _intersect_ts(p1, p2, v1, v2)
            if inter and (0.0 <= t2 <= 1.0) and (f_low <= t1 <= f_high):
                v_cr = p1.add( p2.sub(p1).mul_num(t1) )
                res.append(v_cr)
        if sortreduce:
            return _sortreduce_by_distance(res, p1)  
        return res

    @cython.nonecheck(False)
    cpdef list intersect_line(self, Vec2 p1, Vec2 p2, bint sortreduce=True):
        return self.intersect_general(p1, p2, MINUS_BIG_REAL, BIG_REAL, sortreduce)

    @cython.nonecheck(False)
    cpdef list intersect_ray(self, Vec2 p1, Vec2 p2, bint sortreduce=True):
        return self.intersect_general(p1, p2, 0.0, BIG_REAL, sortreduce)

    @cython.nonecheck(False)
    cpdef list intersect_segment(self, Vec2 p1, Vec2 p2, bint sortreduce=True):
        return self.intersect_general(p1, p2, 0.0, 1.0, sortreduce)



    