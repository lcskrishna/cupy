###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef int Ordering 'hiprandOrdering_t'
    ctypedef int RngType 'hiprandRngType_t'

    ctypedef void* Generator 'hiprandGenerator_t'


###############################################################################
# Enum
###############################################################################

cpdef enum:
    HIPRAND_RNG_PSEUDO_DEFAULT = 100
    HIPRAND_RNG_PSEUDO_XORWOW = 101
    HIPRAND_RNG_PSEUDO_MRG32K3A = 121
    HIPRAND_RNG_PSEUDO_MTGP32 = 141
    HIPRAND_RNG_PSEUDO_MT19937 = 142
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 161
    HIPRAND_RNG_QUASI_DEFAULT = 200
    HIPRAND_RNG_QUASI_SOBOL32 = 201
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202
    HIPRAND_RNG_QUASI_SOBOL64 = 203
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204

    CURAND_ORDERING_PSEUDO_BEST = 100
    CURAND_ORDERING_PSEUDO_DEFAULT = 101
    CURAND_ORDERING_PSEUDO_SEEDED = 102
    CURAND_ORDERING_QUASI_DEFAULT = 201
