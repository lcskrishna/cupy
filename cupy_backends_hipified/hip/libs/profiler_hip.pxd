cdef extern from *:
    ctypedef int OutputMode 'cudaOutputMode_t'


cpdef enum:
    hipKeyValuePair = 0
    hipCSV = 1

cpdef initialize(str config_file, str output_file, int output_mode)
cpdef start()
cpdef stop()
