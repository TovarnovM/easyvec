#cython: embedsignature=True

import numpy as np
from libc.math cimport fabs, sqrt, pi
cimport cython
from .vectors cimport CMP_TOL, real, Vec2, rational, BIG_REAL, MINUS_BIG_REAL
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_GE, Py_GT, Py_NE


cdef array.array _int_array_template = array.array('i', [])
cdef array.array _double_array_template = array.array('d', [])

@cython.binding(True)
@cython.embedsignature(True)
cpdef Vec2 _convert(object candidate):
    """
    my docs
    """
    if isinstance(candidate, Vec2):
        return <Vec2>(candidate)
    else:
        return Vec2(candidate[0], candidate[1])

@cython.embedsignature(True)
def intersect(*args, **kwargs):
    """
    Возвращает точку пересечения двух сущностей (или None, если они не перечекаются).
    В качестве сущностей могут быть бесконечные линии, лучи, отрезки, дуги.
    Сущности задаются двумя точками, через которые они проходят (кроме дуг, они задаются центром, радиусом, и двумя углами).
    К сожалению, пока нельзя найти пересечение двух дуг(
    
    Отрезки задаются кортежем (p1, p2) - двумя крайними точками отрезка. И обозначить их можно именованными аргументами: 
        'segment', 's', 'segment1', 's1', 'segment2', 's2',
    также если аргументы не именованы, то они будут интерпретированы как точки для отрезков. 
    
    Лучи задаются кортежем (p1, p2) - точкой, из которой испускается луч, и точкой, через которую он проходит. 
    И обозначить их можно именованными аргументами: 
        'ray', 'r', 'ray1', 'r1', 'ray2', 'r2'
    
    Бесконечные линии задаются кортежем (p1, p2) - двумя точками, через которые проходит линия. 
    И обозначить их можно именованными аргументами:
        'line', 'l', 'line1', 'l1', 'line2', 'l2'

    Дуга задаются кортежем (ctnter, r, angle_from, angle_to) - центром окружности дуги, радиусом, начальным и конечным углом. 
    И обозначить ее можно именованными аргументами:
        'arc', 'a'


    Примеры использования:
        >>> p_intersect = intersect(p1, p2, p3, p4)                 # p_intersect есть пересечение двух отрезков (p1, p2) и (p3, p4)
        >>> p_intersect = intersect(p1, p2, s=(p3, p4))           # p_intersect есть пересечение двух отрезков (p1, p2) и (p3, p4)
        >>> p_intersect = intersect(p1, p2, segment=(p3, p4))     # p_intersect есть пересечение двух отрезков (p1, p2) и (p3, p4)
        >>> p_intersect = intersect(p1, p2, s2=(p3, p4))          # p_intersect есть пересечение двух отрезков (p1, p2) и (p3, p4)
        
        >>> p_intersect = intersect(s=(p1, p2), s2=(p3, p4))    # p_intersect есть пересечение двух отрезков (p1, p2) и (p3, p4)
        >>> p_intersect = intersect(s1=(p1, p2), s2=(p3, p4))   # p_intersect есть пересечение двух отрезков (p1, p2) и (p3, p4)
        >>> p_intersect = intersect(s=(p1, p2), segment=(p3, p4))# p_intersect есть пересечение двух отрезков (p1, p2) и (p3, p4)

        >>> p_intersect = intersect(p1, p2, ray=(p3, p4))          # p_intersect есть пересечение отрезка (p1, p2) и луча (p3, p4)
        >>> p_intersect = intersect(p1, p2, r=(p3, p4))            # p_intersect есть пересечение отрезка (p1, p2) и луча (p3, p4)
        >>> p_intersect = intersect(p1, p2, ray2=(p3, p4))         # p_intersect есть пересечение отрезка (p1, p2) и луча (p3, p4)
        
        >>> p_intersect = intersect(r1=(p1, p2), r2=(p3, p4))    # p_intersect есть перечечение двух лучей (p1, p2) и (p3, p4)
        >>> p_intersect = intersect(r1=(p1, p2), ray2=(p3, p4))  # p_intersect есть перечечение двух лучей (p1, p2) и (p3, p4)
        >>> p_intersect = intersect(s=(p1, p2), ray2=(p3, p4))   # p_intersect есть перечечение отрезка (p1, p2) и луча (p3, p4)

        >>> p_intersect = intersect(p1, p2, line=(p3, p4))         # p_intersect есть пересечение отрезка (p1, p2) и линии (p3, p4)
        >>> p_intersect = intersect(p1, p2, l=(p3, p4))            # p_intersect есть пересечение отрезка (p1, p2) и линии (p3, p4)
        >>> p_intersect = intersect(p1, p2, l1=(p3, p4))           # p_intersect есть пересечение отрезка (p1, p2) и линии (p3, p4)

        >>> p_intersect = intersect(p1, p2, a=(p3, r, a1, a2))     # p_intersect есть пересечение отрезка (p1, p2) и дуги (p3, r, a1, a2)
        >>> p_intersect = intersect(p1, p2, arc=(p3, r, a1, a2))   # p_intersect есть пересечение отрезка (p1, p2) и дуги (p3, r, a1, a2)
        и т.д.

    В качестве p1, p2, p3, p4 могут быть Vec2, кортежи, списки, массивы.... Всё, что поддерживает индексацию [0] и [1], возвращая при этом числа
    """
    cdef Vec2 u1, u2, v1, v2, center
    cdef real r, start_angle, end_angle
    cdef int alen = len(args)
    cdef int klen = len(kwargs)
    cdef str key1, key2
    cdef object some1, some2, tp1, tp2
    cdef set line_set = {'line', 'l', 'line1', 'l1', 'line2', 'l2'}
    cdef set segment_set = {'segment', 's', 'segment1', 's1', 'segment2', 's2'}
    cdef set ray_set = {'ray', 'r', 'ray1', 'r1', 'ray2', 'r2'}
    cdef set arc_set = {'arc', 'a'}
    
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
            
            if k in line_set:
                v1 = _convert(kwargs[k][0])
                v2 = _convert(kwargs[k][1])
                return intersect_line_segment(v1, v2, u1, u2)
            elif k in segment_set:
                v1 = _convert(kwargs[k][0])
                v2 = _convert(kwargs[k][1])
                return intersect_segments(v1, v2, u1, u2)  
            elif k in ray_set:
                v1 = _convert(kwargs[k][0])
                v2 = _convert(kwargs[k][1])
                return intersect_ray_segment(v1, v2, u1, u2)  
            elif k in arc_set:
                center = _convert(kwargs[k][0])
                r = <real>(kwargs[k][1])
                start_angle = <real>(kwargs[k][2])
                end_angle = <real>(kwargs[k][3])
                return intersect_arc_segment(center, r, start_angle, end_angle, u1, u2)


        raise ValueError(f'Неправильные аргументы {args} {kwargs}')
         
    elif alen == 0 and klen == 2:
        key1, some1= kwargs.popitem()
        key2, some2 = kwargs.popitem()
        if key1 in line_set:
            u1 = _convert(some1[0])
            u2 = _convert(some1[1])
            if key2 in segment_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_line_segment(u1, u2, v1, v2)
            elif key2 in line_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_lines(u1, u2, v1, v2)
            elif key2 in ray_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_ray_line(v1, v2, u1, u2)
            elif key2 in arc_set:
                center = _convert(some2[0])
                r = <real>(some2[1])
                start_angle = <real>(some2[2])
                end_angle = <real>(some2[3])
                return intersect_arc_line(center, r, start_angle, end_angle, u1, u2)
        elif key1 in segment_set:
            u1 = _convert(some1[0])
            u2 = _convert(some1[1])
            if key2 in segment_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_segments(u1, u2, v1, v2)
            elif key2 in line_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_line_segment(v1, v2, u1, u2)
            elif key2 in ray_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_ray_segment(v1, v2, u1, u2) 
            elif key2 in arc_set:
                center = _convert(some2[0])
                r = <real>(some2[1])
                start_angle = <real>(some2[2])
                end_angle = <real>(some2[3])
                return intersect_arc_segment(center, r, start_angle, end_angle, u1, u2)
        elif key1 in ray_set:
            u1 = _convert(some1[0])
            u2 = _convert(some1[1])
            if key2 in segment_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_ray_segment(u1, u2, v1, v2)
            elif key2 in line_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_ray_line(u1, u2, v1, v2)
            elif key2 in ray_set:
                v1 = _convert(some2[0])
                v2 = _convert(some2[1])
                return intersect_rays(v1, v2, u1, u2)
            elif key2 in arc_set:
                center = _convert(some2[0])
                r = <real>(some2[1])
                start_angle = <real>(some2[2])
                end_angle = <real>(some2[3])
                return intersect_arc_ray(center, r, start_angle, end_angle, u1, u2)
    raise ValueError(f'Неправильные аргументы {args} {kwargs}')  


@cython.nonecheck(False)
cpdef bint is_bbox_intersect(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2):
    """
    Проверяет, пересекаются ли два прямоугольника со сторонами, параллельными осям координат.
    """
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
    """
    Возвращает точку пересечения двух бесконечных линий (u1, u2), (v1, v2)
    None, если линии параллельны
    """
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(u1, u2, v1, v2)
    if not suc:
        return None
    return u1.add( u2.sub(u1).mul_num(t1) )

@cython.nonecheck(False)
cpdef Vec2 intersect_segments(Vec2 u1, Vec2 u2, Vec2 v1, Vec2 v2):
    """
    Возвращает точку пересечения двух отрезков (u1, u2), (v1, v2)
    None, если перечечения нет
    """
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
    """
    Возвращает точку пересечения двух лучей (u1, u2), (v1, v2)
    None, если перечечения нет
    """
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(u1, u2, v1, v2)
    if (not suc) or t1 < 0.0 or t2 < 0.0 :
        return None
    return u1.add( u2.sub(u1).mul_num(t1) )

@cython.nonecheck(False)
cpdef Vec2 intersect_ray_line(Vec2 r1, Vec2 r2, Vec2 v1, Vec2 v2):
    """
    Возвращает точку пересечения луча (u1, u2) и бесконечной линии (v1, v2)
    None, если перечечения нет
    """
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(r1, r2, v1, v2)
    if (not suc) or t1 < 0.0:
        return None
    return r1.add( r2.sub(r1).mul_num(t1) )    

@cython.nonecheck(False)
cpdef Vec2 intersect_ray_segment(Vec2 r1, Vec2 r2, Vec2 v1, Vec2 v2):
    """
    Возвращает точку пересечения луча (u1, u2) и отрезка (v1, v2)
    None, если перечечения нет
    """
    cdef:
        bint suc
        real t1, t2
    suc, t1, t2 = _intersect_ts(r1, r2, v1, v2)
    if (not suc) or t1 < 0.0 or t2 < 0.0 or t2 > 1.0 :
        return None
    return r1.add( r2.sub(r1).mul_num(t1) )   

@cython.nonecheck(False)
cpdef Vec2 intersect_line_segment(Vec2 u1, Vec2 u2, Vec2 s1, Vec2 s2):
    """
    Возвращает точку пересечения бесконечной линии (u1, u2) и отрезка (v1, v2)
    None, если перечечения нет
    """
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

   
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef bint is_in_polygon(Vec2 point, list polygon_points):
    """
    Проверяет, находится ли точка внутри полигона
    """
    cdef Vec2 point1 = Vec2(point.x + 1e6, point.y)
    cdef int plen = len(polygon_points)
    cdef int i, n_intersect = 0
    cdef Vec2 v1 = <Vec2>(polygon_points[plen-1])
    cdef Vec2 v2
    cdef:
        bint suc
        real t1, t2
    for i in range(plen):
        v2 = <Vec2>(polygon_points[i])
        if is_bbox_intersect(point, point1, v1, v2):
            suc, t1, t2 = _intersect_ts(point, point1, v1, v2)
            if suc and t1 >= 0.0 and t1 < 1.0 and t2 >= 0.0 and t2 < 1.0 :
                n_intersect += 1
        v1 = v2
    if n_intersect > 0 and n_intersect % 2 != 0:
        return True
    return False

@cython.nonecheck(False)
@cython.cdivision(True)
cpdef real _closest_point_t(Vec2 u1, Vec2 u2, Vec2 p):
    cdef Vec2 u = u2.sub(u1)
    cdef real znam = u.dot(u)
    if fabs(znam) < CMP_TOL:
        return 0.0
    cdef Vec2 v = u1.sub(p)
    return - u.dot(v) / znam

@cython.nonecheck(False)
cpdef Vec2 closest_on_line(Vec2 u1, Vec2 u2, Vec2 p):
    """
    Возвращает ближайшую точку на бесконечной линии (u1, u2) к точке p
    """
    cdef real t = _closest_point_t(u1, u2, p)
    return (u1.mul_num(1 - t)).add(u2.mul_num(t))

@cython.nonecheck(False)
cpdef Vec2 closest_on_ray(Vec2 u1, Vec2 u2, Vec2 p):
    """
    Возвращает ближайшую точку на луче (u1, u2) к точке p
    """
    cdef real t = _closest_point_t(u1, u2, p)
    if t < 0:
        t = 0
    return (u1.mul_num(1 - t)).add(u2.mul_num(t))

@cython.nonecheck(False)
cpdef Vec2 closest_on_segment(Vec2 u1, Vec2 u2, Vec2 p):
    """
    Возвращает ближайшую точку на отрезке (u1, u2) к точке p
    """
    cdef real t = _closest_point_t(u1, u2, p)
    if t < 0:
        t = 0
    elif t > 1:
        t = 1
    return (u1.mul_num(1 - t)).add(u2.mul_num(t))

def closest(*args, **kwargs):
    """
    Возвращает ближайшую точку на сущности к другой, заданной точке
    В качестве сущности могут быть бесконечные линии, лучи, отрезки.
    Сущности задаются двумя точками, через которые они проходят
    
    Отрезки задаются кортежем (p1, p2) - двумя крайними точками отрезка. И обозначить их можно именованными аргументами: 
        'segment', 's', 'segment1', 's1', 'segment2', 's2',
    также если аргументы не именованы, то они будут интерпретированы как точки для отрезков. 
    
    Лучи задаются кортежем (p1, p2) - точкой, из которой испускается луч, и точкой, через которую он проходит. 
    И обозначить их можно именованными аргументами: 
        'ray', 'r', 'ray1', 'r1', 'ray2', 'r2'
    
    Бесконечные линии задаются кортежем (p1, p2) - двумя точками, через которые проходит линия. 
    И обозначить их можно именованными аргументами:
        'line', 'l', 'line1', 'l1', 'line2', 'l2'

    Заданную точку можно обозанчить именованными аргументами:
        'point', 'p'


    Примеры использования:
        >>> p_nearest = closest(p1, p2, p)       # p_nearest есть ближайшая точка к точке "p", и которая принадлежит отрезку (p1, p2)
        >>> p_nearest = closest(p1, p2, p=p)   # p_nearest есть ближайшая точка к точке "p", и которая принадлежит отрезку (p1, p2)
        >>> p_nearest = closest(p1, p2, point=p)         # p_nearest есть ближайшая точка к точке "p", и которая принадлежит отрезку (p1, p2)
        >>> p_nearest = closest(s=(p1, p2), p=p)       # p_nearest есть ближайшая точка к точке "p", и которая принадлежит отрезку (p1, p2)
        >>> p_nearest = closest(segment=(p1, p2), p=p) # p_nearest есть ближайшая точка к точке "p", и которая принадлежит отрезку (p1, p2)

        >>> p_nearest = closest(r=(p1, p2), p=p)       # p_nearest есть ближайшая точка к точке "p", и которая принадлежит лучу (p1, p2)
        >>> p_nearest = closest(ray=(p1, p2), p=p)     # p_nearest есть ближайшая точка к точке "p", и которая принадлежит лучу (p1, p2)

        >>> p_nearest = closest(line=(p1, p2), p=p)     # p_nearest есть ближайшая точка к точке "p", и которая принадлежит линии (p1, p2)
        и т.д.

    В качестве p1, p2, p могут быть Vec2, кортежи, списки, массивы.... Всё, что поддерживает индексацию [0] и [1], возвращая при этом числа
    """
    cdef Vec2 u1, u2, p
    cdef int alen = len(args)
    cdef int klen = len(kwargs)
    cdef str key1, key2
    cdef object some1, some2
    cdef set line_set = {'line', 'l', 'line1', 'l1', 'line2', 'l2'}
    cdef set segment_set = {'segment', 's', 'segment1', 's1', 'segment2', 's2'}
    cdef set ray_set = {'ray', 'r', 'ray1', 'r1', 'ray2', 'r2'}
    cdef set point_set = {'point', 'p'}
    
    if alen == 3 and klen == 0:
        u1 = _convert(args[0])
        u2 = _convert(args[1])
        p = _convert(args[2])
        return closest_on_segment(u1, u2, p)
    elif alen == 2 and klen == 1:
        u1 = _convert(args[0])
        u2 = _convert(args[1])
        key1, some1 = kwargs.popitem()
        p = _convert(some1)
        if key1 in point_set:
            return closest_on_segment(u1, u2, p)           

        raise ValueError(f'Неправильные аргументы {args} {kwargs}')
    elif alen == 1 and klen == 1:
        p = _convert(args[0])
        key1, (some1, some2) = kwargs.popitem()
        u1 = _convert(some1)
        u2 = _convert(some2)
        if key1 in line_set:
            return closest_on_line(u1, u2, p)
        elif key1 in ray_set:
            return closest_on_ray(u1, u2, p)
        elif key1 in segment_set:
            return closest_on_segment(u1, u2, p)
         
    elif alen == 0 and klen == 2:
        key1, some1 = kwargs.popitem()
        key2, some2 = kwargs.popitem()
        if key1 in point_set:
            p = _convert(some1)
            u1 = _convert(some2[0])
            u2 = _convert(some2[1])
            if key2 in segment_set:
                return closest_on_segment(u1, u2, p)
            elif key2 in line_set:
                return closest_on_line(u1, u2, p)
            elif key2 in ray_set:
                return closest_on_ray(u1, u2, p)
        elif key1 in line_set:
            u1 = _convert(some1[0])
            u2 = _convert(some1[1])
            if key2 in point_set:
                p = _convert(some2)
                return closest_on_line(u1, u2, p)
        elif key1 in ray_set:
            u1 = _convert(some1[0])
            u2 = _convert(some1[1])
            if key2 in point_set:
                p = _convert(some2)
                return closest_on_ray(u1, u2, p)
        elif key1 in segment_set:
            u1 = _convert(some1[0])
            u2 = _convert(some1[1])
            if key2 in point_set:
                p = _convert(some2)
                return closest_on_segment(u1, u2, p)
    raise ValueError(f'Неправильные аргументы {args} {kwargs}') 


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef (bint, real, real) _intersect_circle_line_ts(Vec2 center, real r, Vec2 u1, Vec2 u2):
    cdef Vec2 v = u2.sub(u1)
    cdef real a = v.len_sqared() # a * x^2 + b * x + c
    if a < CMP_TOL:
        return (False, 0, 0)
    cdef Vec2 u = u1.sub(center)
    cdef real b = 2 * u.dot(v)
    cdef real c = u.len_sqared() - r * r
    cdef real D = b*b - 4*a*c  # determenant
    cdef real t1, t2
    if D < 0:
        return (False, 0, 0)
    elif D < CMP_TOL:
        t1 = -b/(2*a) 
        t2 = t1
        return (True, t1, t2)
    else:
        D = sqrt(D)
        t1 = (-b + D)/(2*a)
        t2 = (-b - D)/(2*a)
        return (True, t1, t2)

@cython.nonecheck(False)
cdef inline real normalize_angle2pi(real angle) nogil:
    """
    Нормализвут угол. Приводит его к виду  0 <= angle <= 2*pi
    """
    while angle >= 2*pi:
        angle -= 2*pi
    while angle < 0:
        angle += 2*pi
    return angle


@cython.nonecheck(False)
cdef inline bint _angle_between(real start, real end, real mid) nogil:
    start = normalize_angle2pi(start)
    end   = normalize_angle2pi(end)
    end = end - start + 2*pi if (end - start) < 0 else end - start
    mid = normalize_angle2pi(mid)
    mid = mid - start + 2*pi if (mid - start) < 0 else mid - start
    return mid <= end


@cython.nonecheck(False)
cpdef bint angle_between(real start, real end, real mid):
    """
    Проверяет лежит ли луч, выходящий из начала координат под углом mid, внутри угла, образаванного двумя лучами,
    выходящими из начала координат под углами start и end. Область внутри угла образована вращением луча start до луча end против часовой стрелки 
    """
    return _angle_between(start, end, mid)
        

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef list intersect_arc_line(Vec2 center, real r, real start_angle, real end_angle, Vec2 u1, Vec2 u2):
    """
    Возвращает списко точек перечечения дуги (center, r, start_angle, end_angle) и линии (u1, u2). Пустой список, в случае, когда пересечений нет.
    Дуга задаются center, r, start_angle, end_angle - центром окружности дуги, радиусом, начальным и конечным углом. 

    """
    cdef list res = []
    cdef bint is_intersect
    cdef real t1, t2, angle_t1, angle_t2
    cdef Vec2 p1, p2
    is_intersect, t1, t2 = _intersect_circle_line_ts(center, r, u1, u2)
    if not is_intersect:
        return res
    p1 = (u1.mul_num(1 - t1)).add(u2.mul_num(t1))
    angle_t1 = (p1.sub(center)).angle_to_xy(1, 0)
    
    if _angle_between(start_angle, end_angle, angle_t1):
        res.append(p1)

    if fabs(t1 - t2) < CMP_TOL:
        return res

    p2 = (u1.mul_num(1 - t2)).add(u2.mul_num(t2))
    angle_t2 = (p2.sub(center)).angle_to_xy(1, 0)
    if _angle_between(start_angle, end_angle, angle_t2):
        res.append(p2)
    return res

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef list intersect_arc_ray(Vec2 center, real r, real start_angle, real end_angle, Vec2 u1, Vec2 u2):
    """
    Возвращает списко точек перечечения дуги (center, r, start_angle, end_angle) и луча (u1, u2). Пустой список, в случае, когда пересечений нет.
    Дуга задаются center, r, start_angle, end_angle - центром окружности дуги, радиусом, начальным и конечным углом. 

    """
    cdef list res = []
    cdef bint is_intersect
    cdef real t1, t2, angle_t1, angle_t2
    cdef Vec2 p1, p2
    is_intersect, t1, t2 = _intersect_circle_line_ts(center, r, u1, u2)
    if not is_intersect:
        return res

    if t1 >= 0:
        p1 = (u1.mul_num(1 - t1)).add(u2.mul_num(t1))
        angle_t1 = (p1.sub(center)).angle_to_xy(1, 0)
        
        if _angle_between(start_angle, end_angle, angle_t1):
            res.append(p1)

    if fabs(t1 - t2) < CMP_TOL:
        return res

    if t2 >= 0:
        p2 = (u1.mul_num(1 - t2)).add(u2.mul_num(t2))
        angle_t2 = (p2.sub(center)).angle_to_xy(1, 0)
        if _angle_between(start_angle, end_angle, angle_t2):
            res.append(p2)
    return res    

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef list intersect_arc_segment(Vec2 center, real r, real start_angle, real end_angle, Vec2 u1, Vec2 u2):
    """
    Возвращает списко точек перечечения дуги (center, r, start_angle, end_angle) и отрезка (u1, u2). Пустой список, в случае, когда пересечений нет.
    Дуга задаются center, r, start_angle, end_angle - центром окружности дуги, радиусом, начальным и конечным углом. 

    """
    cdef list res = []
    cdef bint is_intersect
    cdef real t1, t2, angle_t1, angle_t2
    cdef Vec2 p1, p2
    is_intersect, t1, t2 = _intersect_circle_line_ts(center, r, u1, u2)
    if not is_intersect:
        return res

    if 0 <= t1 <= 1:
        p1 = (u1.mul_num(1 - t1)).add(u2.mul_num(t1))
        angle_t1 = (p1.sub(center)).angle_to_xy(1, 0)
        
        if _angle_between(start_angle, end_angle, angle_t1):
            res.append(p1)

    if fabs(t1 - t2) < CMP_TOL:
        return res

    if 0 <= t2 <= 1:
        p2 = (u1.mul_num(1 - t2)).add(u2.mul_num(t2))
        angle_t2 = (p2.sub(center)).angle_to_xy(1, 0)
        if _angle_between(start_angle, end_angle, angle_t2):
            res.append(p2)
    return res 


@cython.final
cdef class Rect:
    """
    Класс, представляющий прямоугольник со сторонами, параллельными осям координат (AABB - англ. термин)

    Содержит 4 поля с float числами: x1, y1, x2, y2. Причем точка (x1, y1) - нижняя-левая, (x2, y2) - верхняя правая

    """
    @classmethod
    def null(cls) -> Rect:
        """
        Возвращает "пустой" прямоугольник (0,0,0,0)
        """
        return cls(0,0,0,0)

    @classmethod
    def from_dict(cls, dct: dict) -> Rect:
        """
        Создает прямоугольник (x1,y1,x2,y2) из словаря, у которого есть ключи 'x1','y1','x2','y2'
        """    
        return cls(dct['x1'], dct['y1'], dct['x2'], dct['y2'])

    @classmethod
    def bbox(cls, *args):
        """
        Создает прямоугольник, описанный вокруг множества точек
        """
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
        """
        return np.array([self.x1, self.y1, self.x2, self.y2])
        """
        return np.array([self.x1, self.y1, self.x2, self.y2])

    cpdef tuple as_tuple(self):
        """
        return (self.x1, self.y1, self.x2, self.y2)
        """
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
        """
        Проверяет, пустой ли прямоугольник == (0,0,0,0)
        """
        return fabs(self.x1) < CMP_TOL and fabs(self.y1) < CMP_TOL and \
               fabs(self.x2) < CMP_TOL and fabs(self.y2) < CMP_TOL

    @cython.nonecheck(False)
    cpdef bint is_in(self, Vec2 p):
        """
        Проверяет, находится ли точка p внутри прямоугольника
        """    
        return (self.x1 <= p.x <= self.x2) and (self.y1 <= p.y <= self.y2)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef Vec2 intersect_general(self, Vec2 p1, Vec2 p2, real f_low, real f_high,bint ret_closest=True):
        """
        Функция возвращает точку пересечения прямоугольника и сущности (линия, луч, отрезок)
        """   
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
        """
        Функция возвращает точку пересечения прямоугольника и отрезка (ближайшую к p1)
        """   
        return self.intersect_general(p1, p2, 0.0, 1.0, ret_closest)

    @cython.nonecheck(False)
    cpdef Vec2 intersect_ray(self, Vec2 p1, Vec2 p2, bint ret_closest=True):
        """
        Функция возвращает точку пересечения прямоугольника и луча (ближайшую к p1)
        """  
        return self.intersect_general(p1, p2, 0.0, BIG_REAL, ret_closest)

    @cython.nonecheck(False)
    cpdef Vec2 intersect_line(self, Vec2 p1, Vec2 p2, bint ret_closest=True):
        """
        Функция возвращает точку пересечения прямоугольника и линии (ближайшую к p1)
        """  
        return self.intersect_general(p1, p2, MINUS_BIG_REAL, BIG_REAL, ret_closest)

    cpdef real area(self):
        """
        Функция возвращает площадь прямоугольника
        """  
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    cpdef real perimeter(self):
        """
        Функция возвращает периметр
        """  
        return 2*(self.x2 - self.x1 + self.y2 - self.y1)
    
    @cython.nonecheck(False)
    cpdef bint is_intersect_rect(self, Rect other):
        """
        Проверяет, переcекает ли другой прямоугольник
        """  
        return (self.x1 <= other.x2) and (self.x2 >= other.x1) \
           and (self.y1 <= other.y2) and (self.y2 >= other.y1)

    @cython.nonecheck(False)
    cpdef Rect intersect_rect(self, Rect other):
        """
        Возвращает общий прямоугольник (пересечение с другим прямоугольником)
        """ 
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
        """
        Возвращает прямоугольник, включающий в себя исходный и other прямоугольники (объединение прямоугольников)
        """ 
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
        """
        Возвращает прямоугольник, включающий в себя исходный прямоугольник и точку vec
        """ 
        return self.union_point(vec.x, vec.y)

    @cython.nonecheck(False)
    cpdef Rect union_point(self, real x, real y):
        """
        Возвращает прямоугольник, включающий в себя исходный прямоугольник и точку (x, y)
        """ 
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
        """
        Возвращает точку пересечения прямоугольника и сущности (или None, если они не перечекаются).
        В качестве сущности могут быть бесконечные линия, луч, отрезок или другой прямоугольник.
        Сущности задаются двумя точками, через которые они проходят (кроме прямоугольника).
        
        Отрезки задаются кортежем (p1, p2) - двумя крайними точками отрезка. И обозначить их можно именованными аргументами: 
            'segment', 's',
        также если аргументы не именованы, то они будут интерпретированы как точки для отрезков. 
        
        Лучи задаются кортежем (p1, p2) - точкой, из которой испускается луч, и точкой, через которую он проходит. 
        И обозначить их можно именованными аргументами: 
            'ray', 'r'
        
        Бесконечные линии задаются кортежем (p1, p2) - двумя точками, через которые проходит линия. 
        И обозначить их можно именованными аргументами:
            'line', 'l'


        Примеры использования:
            >>> rect1.intersect(rect2)    # пересечение двух прямоугольников
            >>> rect1.intersect((p1,p2))  # пересечение c отрезком (p1,p2)
            >>> rect1.intersect(p1,p2)    # пересечение c отрезком (p1,p2)
            >>> rect1.intersect((p1,p2))  # пересечение c отрезком (p1,p2)
            >>> rect1.intersect(s=(p1,p2))  # пересечение c отрезком (p1,p2)
            >>> rect1.intersect(r=(p1,p2))  # пересечение c лучом (p1,p2)
            >>> rect1.intersect(line=(p1,p2))  # пересечение c линией (p1,p2)
            и т.д.

        В качестве p1, p2 могут быть Vec2, кортежи, списки, массивы.... Всё, что поддерживает индексацию [0] и [1], возвращая при этом числа
        """
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
        """
        Проверяет, есть ли пересечение с описанным вокруг отрезка (p1, p2) прямоугольником
        """
        return (self.x1 <= fmax(p1.x, p2.x)) and (self.x2 >= fmin(p1.x, p2.x)) \
           and (self.y1 <= fmax(p1.y, p2.y)) and (self.y2 >= fmin(p1.y, p2.y))


@cython.final
cdef class PolyLine:
    """
    Класс полигона или полилинии
    
    Поля:
        vecs - list[Vec2] - это список из точек
        enclosed - bool - является ли полилиния замкнутой
        bbox - Rect  - описанный прямоугольник
    """
    @classmethod
    def from_dict(cls, dct):
        """
        Созадть полигон/полилинию из словаря вида
        {
            'vecs': [{'x': 1, 'y': 2}, {'x':3, 'y':4}],
            'enclosed': False
        }
        """
        vecs = [Vec2.from_dict(vd) for vd in dct['vecs']]
        enclosed = dct['enclosed']
        return cls(vecs, enclosed)

    def __cinit__(self, vecs: list, enclosed=True, copy_data=False):
        """
        Созадть полигон/полилинию
        vecs - список из точек Vec2
        enclosed - является ли замкнутым полилиния
        copy_data - нужно ли копировать элементы списка, или просто сослаться на него (в случае True, 
            в качестве точек могут быть использованы ec2, кортежи, списки, массивы.... Всё, что поддерживает индексацию [0] и [1], возвращая при этом числа)
        """
        cdef int vec_len, i
        cdef Vec2 tmp
        if copy_data or (not isinstance(vecs[0], Vec2)):
            self.vecs = []
            vec_len = len(vecs)
            for i in range(vec_len):
                if isinstance(vecs[i], Vec2):
                    tmp = (<Vec2>(vecs[i])).copy()
                else:
                    tmp = Vec2(vecs[i][0], vecs[i][1])
                self.vecs.append( tmp )
        else:
            self.vecs = vecs

        self.vlen = len(self.vecs)
        if self.vlen < 2:
            raise ValueError(f'Слишком мало точек для линии. Необходимо больше, чем 1')
        self.enclosed = enclosed
        self.bbox = Rect.bbox(self.vecs)

    @cython.nonecheck(False)
    cpdef PolyLine copy(self):
        return PolyLine(self.vecs, self.enclosed, copy_data=True)

    @cython.nonecheck(False)
    cpdef PolyLine clone(self):
        return PolyLine(self.vecs, self.enclosed, copy_data=True)

    def to_dict(self):
        """
        return {
            'vecs': [v.to_dict() for v in self.vecs],
            'enclosed': bool(self.enclosed)
        }
        """
        return {
            'vecs': [v.to_dict() for v in self.vecs],
            'enclosed': bool(self.enclosed)
        }

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
        """
        Функция возвращает точки пересечения прямоугольника и сущности (линия, луч, отрезок)
        """   
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
        """
        Функция возвращает точки пересечения прямоугольника и линии
        sortreduce - нужно ли сортировать точки по расстоянию от p1
        """   
        return self.intersect_general(p1, p2, MINUS_BIG_REAL, BIG_REAL, sortreduce)

    @cython.nonecheck(False)
    cpdef list intersect_ray(self, Vec2 p1, Vec2 p2, bint sortreduce=True):
        """
        Функция возвращает точки пересечения прямоугольника и луча
        sortreduce - нужно ли сортировать точки по расстоянию от p1
        """   
        return self.intersect_general(p1, p2, 0.0, BIG_REAL, sortreduce)

    @cython.nonecheck(False)
    cpdef list intersect_segment(self, Vec2 p1, Vec2 p2, bint sortreduce=True):
        """
        Функция возвращает точки пересечения прямоугольника и отрезка
        sortreduce - нужно ли сортировать точки по расстоянию от p1
        """   
        return self.intersect_general(p1, p2, 0.0, 1.0, sortreduce)

    @cython.nonecheck(False)
    cpdef bint is_in(self, Vec2 point):
        """
        Лежит ли точка внутри полигона
        """   
        if not self.bbox.is_in(point):
            return False
        return is_in_polygon(point, self.vecs)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PolyLine transform(self, Mat2 m):
        """
        Создает новую полилинию, точки которой являются произведением исходных точек с матрицей Mat2 m
        """   
        cdef list vecs = []
        cdef int i

        for i in range(self.vlen):
            vecs.append(m.mul_vec(<Vec2>(self.vecs[i])))

        return PolyLine(vecs, self.enclosed, copy_data=False)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PolyLine add_vec(self, Vec2 v):
        """
        Создает новую полилинию, точки которой являются смещением исходных точек на вектор v
        """   
        cdef list vecs = []
        cdef int i

        for i in range(self.vlen):
            vecs.append(v.add_vec(<Vec2>(self.vecs[i])))

        return PolyLine(vecs, self.enclosed, copy_data=False)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef real get_area(self, bint always_positive=True):
        """
        Получить площать полигона
        """   
        cdef:
            real res = 0
            int i
            Vec2 vi, vi1

        vi = self.vecs[self.vlen-1]
        for i in range(self.vlen):
            vi1 = self.vecs[i]
            res += vi.cross(vi1)
            vi = vi1
        if always_positive:
            res = fabs(res)
        return res/2

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Vec2 get_center_mass(self):
        """
        Получить координату ц.м. полигона (полигон считается с равномерной по полщади плотностью)
        """   
        cdef:
            real a = 0
            int i
            Vec2 vi, vi1
            real cx=0, cy=0
            real cross
        vi = self.vecs[self.vlen-1]
        for i in range(self.vlen):
            vi1 = self.vecs[i]
            cross = vi.cross(vi1)
            cx += (vi.x + vi1.x) * cross
            cy += (vi.y + vi1.y) * cross
            a += cross
            vi = vi1
        a /= 2
        cx /= 6*a
        cy /= 6*a
        return Vec2(cx, cy)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef real get_Iz(self, Vec2 z_point):
        """
        Получить момент инерции относительно оси, проходящей через z_point и направленной перпендикулярно плоскости xy 
        (полигон считается с равномерной по полщади плотностью и массой = 1)
        """   
        cdef:
            real znam = 0
            int i
            Vec2 vi, vi1
            real chisl=0
            real cross
        vi = (self.vecs[self.vlen-1]).sub(z_point)
        for i in range(self.vlen):
            vi1 = (self.vecs[i]).sub(z_point)
            cross = fabs(vi.cross(vi1))
            znam += cross
            chisl += cross * (vi.len_sqared() + vi.dot(vi1) + vi1.len_sqared())
            vi = vi1
        return chisl / (6*znam)
    
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint is_selfintersect(self):
        """
        Пересекает ли полигон сам себя
        """           
        cdef:
            int i, j
            Vec2 u1, u2, v1, v2
            bint inter
            real t1, t2

        u1 = self.vecs[self.vlen-1]
        u2 = self.vecs[0]
        for i in range(self.vlen-1):
            for j in range(i+1, self.vlen-1):
                v1 = self.vecs[j]
                v2 = self.vecs[j+1]
                inter, t1, t2 = _intersect_ts(u1, u2, v1, v2)
                if inter and (0<t1<1) and (0<t2<1):
                    return True
            u1, u2 = self.vecs[i], self.vecs[i+1]
        return False

                
