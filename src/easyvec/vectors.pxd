cimport cython

ctypedef double real

ctypedef fused rational:
    cython.short
    cython.int
    cython.long
    cython.float
    cython.double


cdef class Vec2:
    cdef real x, y

    cpdef Vec2 add_num_(self, real num)
    cpdef Vec2 add_num(self, real num)
    cpdef Vec2 add_vec_(self, Vec2 vec)
    cpdef Vec2 add_vec(self, Vec2 vec)
    cpdef Vec2 add_xy_(self, real x, real y)
    cpdef Vec2 add_xy(self, real x, real y)
    cpdef Vec2 add_tup_(self, tuple tup)
    cpdef Vec2 add_tup(self, tuple tup)
    cpdef Vec2 add_list_(self, list tup)
    cpdef Vec2 add_list(self, list tup)
    cpdef Vec2 add_arr_(self, rational[:] arr)
    cpdef Vec2 add_arr(self, rational[:] tup)
    cpdef list keys(self)

    cpdef Vec2 sub_num_(self, real num)
    cpdef Vec2 sub_num(self, real num)
    cpdef Vec2 sub_vec_(self, Vec2 vec)
    cpdef Vec2 sub_vec(self, Vec2 vec)
    cpdef Vec2 sub_xy_(self, real x, real y)
    cpdef Vec2 sub_xy(self, real x, real y)
    cpdef Vec2 sub_tup_(self, tuple tup)
    cpdef Vec2 sub_tup(self, tuple tup)
    cpdef Vec2 sub_list_(self, list tup)
    cpdef Vec2 sub_list(self, list tup)
    cpdef Vec2 sub_arr_(self, rational[:] arr)
    cpdef Vec2 sub_arr(self, rational[:] tup)

    cpdef real[:] as_np(self)
    cpdef (real, real) as_tuple(self)
    cpdef list keys(self)
    cpdef Vec2 neg_(self)

    cpdef Vec2 mul_num_(self, real num)
    cpdef Vec2 mul_num(self, real num)
    cpdef Vec2 mul_vec_(self, Vec2 vec)
    cpdef Vec2 mul_vec(self, Vec2 vec)
    cpdef Vec2 mul_xy_(self, real x, real y)
    cpdef Vec2 mul_xy(self, real x, real y)
    cpdef Vec2 mul_tup_(self, tuple tup)
    cpdef Vec2 mul_tup(self, tuple tup)
    cpdef Vec2 mul_list_(self, list tup)
    cpdef Vec2 mul_list(self, list tup)
    cpdef Vec2 mul_arr_(self, rational[:] arr)
    cpdef Vec2 mul_arr(self, rational[:] tup)

    cpdef real dot(self, Vec2 vec)
    cpdef real dot_xy(self, real x, real y)
    cpdef real dot_tup(self, tuple tup)
    cpdef real dot_list(self, list tup)
    cpdef real dot_arr(self, rational[:] tup)

    cpdef Vec2 div_num_(self, real num)
    cpdef Vec2 div_num(self, real num)
    cpdef Vec2 div_vec_(self, Vec2 vec)
    cpdef Vec2 div_vec(self, Vec2 vec)
    cpdef Vec2 div_xy_(self, real x, real y)
    cpdef Vec2 div_xy(self, real x, real y)
    cpdef Vec2 div_tup_(self, tuple tup)
    cpdef Vec2 div_tup(self, tuple tup)
    cpdef Vec2 div_list_(self, list tup)
    cpdef Vec2 div_list(self, list tup)
    cpdef Vec2 div_arr_(self, rational[:] arr)
    cpdef Vec2 div_arr(self, rational[:] tup)

    cpdef Vec2 floordiv_num_(self, real num)
    cpdef Vec2 floordiv_num(self, real num)
    cpdef Vec2 floordiv_vec_(self, Vec2 vec)
    cpdef Vec2 floordiv_vec(self, Vec2 vec)
    cpdef Vec2 floordiv_xy_(self, real x, real y)
    cpdef Vec2 floordiv_xy(self, real x, real y)
    cpdef Vec2 floordiv_tup_(self, tuple tup)
    cpdef Vec2 floordiv_tup(self, tuple tup)
    cpdef Vec2 floordiv_list_(self, list tup)
    cpdef Vec2 floordiv_list(self, list tup)
    cpdef Vec2 floordiv_arr_(self, rational[:] arr)
    cpdef Vec2 floordiv_arr(self, rational[:] tup)

    cpdef Vec2 mod_num_(self, real num)
    cpdef Vec2 mod_num(self, real num)
    cpdef Vec2 mod_vec_(self, Vec2 vec)
    cpdef Vec2 mod_vec(self, Vec2 vec)
    cpdef Vec2 mod_xy_(self, real x, real y)
    cpdef Vec2 mod_xy(self, real x, real y)
    cpdef Vec2 mod_tup_(self, tuple tup)
    cpdef Vec2 mod_tup(self, tuple tup)
    cpdef Vec2 mod_list_(self, list tup)
    cpdef Vec2 mod_list(self, list tup)
    cpdef Vec2 mod_arr_(self, rational[:] arr)
    cpdef Vec2 mod_arr(self, rational[:] tup)

    cpdef real len(self)
    cpdef real len_sqared(self)
    cpdef Vec2 abs_(self)
    cpdef Vec2 abs(self)
    cpdef Vec2 norm_(self)
    cpdef Vec2 norm(self)
    cpdef Vec2 round_(self, int ndigits=*)
    cpdef Vec2 round(self, int ndigits=*)
    cpdef Vec2 ceil_(self, int ndigits=*)
    cpdef Vec2 ceil(self, int ndigits=*)
    cpdef Vec2 floor_(self, int ndigits=*)
    cpdef Vec2 floor(self, int ndigits=*)
    cpdef Vec2 trunc_(self, int ndigits=*)
    cpdef Vec2 trunc(self, int ndigits=*)

    cpdef real cross(self, Vec2 right)
    cpdef real cross_xy(self, real x, real y)

    cpdef real angle_to_xy(self, real x, real y, int degree=*)
    cpdef real angle_to(self, Vec2 vec, int degree=*)

    cpdef Vec2 rotate90_(self)
    cpdef Vec2 rotate90(self)

    cpdef Vec2 rotate_(self, real angle, int degrees=*)
    cpdef Vec2 rotate(self, real angle, int degrees=*)


cdef class Vec3:
    cdef real x, y, z