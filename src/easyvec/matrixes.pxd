from .vectors cimport Vec2, real, rational, Vec3

cpdef Vec3 _convert(object candidate)

cdef class Mat2:
    cdef public real m11, m12, m21, m22

    cpdef Mat2 copy(self)
    cpdef Mat2 clone(self)
    cpdef Vec2 i_axis(self)
    cpdef Vec2 j_axis(self)
    cpdef Vec2 x_axis(self)
    cpdef Vec2 y_axis(self)
    cpdef Mat2 transpose(self)
    cpdef real det(self)
    cpdef Mat2 inverse(self)
    cpdef Mat2 mul_mat_elements_(self, Mat2 right)
    cpdef Mat2 mul_mat_elements(self, Mat2 right)
    cpdef Mat2 mul_mat_(self, Mat2 right)
    cpdef Mat2 mul_mat(self, Mat2 right)
    cpdef Vec2 mul_vec(self, Vec2 vec)
    cpdef Mat2 mul_num_(self, real num)
    cpdef Mat2 mul_num(self, real num)
    cpdef Mat2 add_num_(self, real num)
    cpdef Mat2 add_num(self, real num)
    cpdef Mat2 add_mat_(self, Mat2 mat)
    cpdef Mat2 add_mat(self, Mat2 mat)
    cpdef Mat2 neg_(self)
    cpdef Mat2 neg(self)
    cpdef list keys(self)
    cpdef real[:,:] as_np(self)
    cpdef tuple as_tuple(self)
    cpdef Mat2 sub_num_(self, real num)
    cpdef Mat2 sub_num(self, real num)
    cpdef Mat2 sub_mat_(self, Mat2 mat)
    cpdef Mat2 sub_mat(self, Mat2 mat)
    cpdef Mat2 div_num_(self, real num)
    cpdef Mat2 div_num(self, real num)
    cpdef Mat2 div_mat_(self, Mat2 mat)
    cpdef Mat2 div_mat(self, Mat2 mat)

cdef class Mat3:
    cdef public real m11, m12, m13
    cdef public real m21, m22, m23
    cdef public real m31, m32, m33
    
    cpdef Mat3 copy(self)
    cpdef Mat3 clone(self)
    cpdef real[:,:] as_np(self)
    cpdef list as_list(self)
    cpdef tuple as_tuple(self)
    cpdef Vec3 i_axis(self)
    cpdef Vec3 j_axis(self)
    cpdef Vec3 k_axis(self)
    cpdef Vec3 x_axis(self)
    cpdef Vec3 y_axis(self)
    cpdef Vec3 z_axis(self)
    cpdef Mat3 transpose(self)
    cpdef Mat3 transpose_(self)
    cpdef real det(self)
    cpdef Mat3 inverse(self)
    cpdef Mat3 mul_mat_elements_(self, Mat3 right)
    cpdef Mat3 mul_mat_elements(self, Mat3 right)
    cpdef Mat3 mul_mat_(self, Mat3 b)
    cpdef Mat3 mul_mat(self, Mat3 b)
    cpdef Vec3 mul_vec(self, Vec3 vec)
    cpdef Mat3 mul_num_(self, real num)
    cpdef Mat3 mul_num(self, real num)
    cpdef Mat3 add_num_(self, real num)
    cpdef Mat3 add_num(self, real num)
    cpdef Mat3 add_mat_(self, Mat3 mat)
    cpdef Mat3 add_mat(self, Mat3 mat)
    cpdef Mat3 neg_(self)
    cpdef Mat3 neg(self)
    cpdef list keys(self)
    cpdef Mat3 sub_num_(self, real num)
    cpdef Mat3 sub_num(self, real num)
    cpdef Mat3 sub_mat_(self, Mat3 mat)
    cpdef Mat3 sub_mat(self, Mat3 mat)
    cpdef Mat3 div_num_(self, real num)
    cpdef Mat3 div_num(self, real num)
    cpdef Mat3 div_mat_(self, Mat3 mat)
    cpdef Mat3 div_mat(self, Mat3 mat)

cpdef Quat _convert_quat(object candidate)

cdef class Quat:
    cdef public real w, x ,y, z

    cpdef Quat copy(self)
    cpdef Quat clone(self)
    cpdef tuple as_tuple(self)
    cpdef list keys(self)
    cpdef real get_modulus(self)
    cpdef real len(self)
    cpdef real get_modulus_squared(self)
    cpdef real len_squared(self)
    cpdef bint is_eq(self, Quat other)
    cpdef Quat mul_quat_(self, Quat right)
    cpdef Quat mul_quat(self, Quat right)
    cpdef Vec3 mul_vec(self, Vec3 vec)
    cpdef Quat mul_num_(self, real num)
    cpdef Quat mul_num(self, real num)
    cpdef Vec3 get_axis(self, bint normalize=*)
    cpdef void set_axis(self, Vec3 axis)
    cpdef real get_angle(self, bint degrees=*)
    cpdef void set_angle(self, real angle, bint degrees=*)
    cpdef Quat conjugate(self)
    cpdef Quat conjugate_(self)
    cpdef Quat inverse_(self)
    cpdef Quat inverse(self)
    cpdef Quat norm_(self, bint raise_zero_len_error=*)
    cpdef Quat norm(self, bint raise_zero_len_error=*)
    cpdef Quat add_num_(self, real num)
    cpdef Quat add_num(self, real num)
    cpdef Quat add_quat_(self, Quat q)
    cpdef Quat add_quat(self, Quat q)
    cpdef Quat sub_num_(self, real num)
    cpdef Quat sub_num(self, real num)
    cpdef Quat sub_quat_(self, Quat q)
    cpdef Quat sub_quat(self, Quat q)
    cpdef Quat div_num_(self, real num)
    cpdef Quat div_num(self, real num)
    cpdef Quat div_quat_(self, Quat q)
    cpdef Quat div_quat(self, Quat q)
    cpdef Quat neg_(self)
    cpdef Quat neg(self)
    cpdef real dot(self, Quat q)
    cpdef Mat3 to_Mat3(self)



