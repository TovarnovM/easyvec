from .vectors cimport Vec2, real, rational


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





