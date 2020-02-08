cdef class Vec3(object):
    def __cinit__(self, x: double, y: double, z: double):
        self.x = x
        self.y = y
        self.z = z 