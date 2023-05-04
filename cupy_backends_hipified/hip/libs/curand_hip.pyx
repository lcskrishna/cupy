# distutils: language = c++

"""Thin wrapper of cuRAND."""
cimport cython  # NOQA

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_rand.h' nogil:
    ctypedef void* Stream 'hipStream_t'

    # Generator
    int hiprandCreateGenerator(Generator* generator, int rng_type)
    int hiprandDestroyGenerator(Generator generator)
    int hiprandGetVersion(int* version)

    # Stream
    int hiprandSetStream(Generator generator, Stream stream)
    int hiprandSetPseudoRandomGeneratorSeed(
        Generator generator, unsigned long long seed)
    int hiprandSetGeneratorOffset(
        Generator generator, unsigned long long offset)
    int hiprandSetGeneratorOrdering(Generator generator, Ordering order)

    # Generation functions
    int hiprandGenerate(
        Generator generator, unsigned int* outputPtr, size_t num)
    int hiprandGenerateLongLong(
        Generator generator, unsigned long long* outputPtr, size_t num)
    int hiprandGenerateUniform(
        Generator generator, float* outputPtr, size_t num)
    int hiprandGenerateUniformDouble(
        Generator generator, double* outputPtr, size_t num)
    int hiprandGenerateNormal(
        Generator generator, float* outputPtr, size_t num,
        float mean, float stddev)
    int hiprandGenerateNormalDouble(
        Generator generator, double* outputPtr, size_t n,
        double mean, double stddev)
    int hiprandGenerateLogNormal(
        Generator generator, float* outputPtr, size_t n,
        float mean, float stddev)
    int hiprandGenerateLogNormalDouble(
        Generator generator, double* outputPtr, size_t n,
        double mean, double stddev)
    int hiprandGeneratePoisson(
        Generator generator, unsigned int* outputPtr, size_t n, double lam)


###############################################################################
# Error handling
###############################################################################

STATUS = {
    0: 'HIPRAND_STATUS_SUCCESS',
    100: 'HIPRAND_STATUS_VERSION_MISMATCH',
    101: 'HIPRAND_STATUS_NOT_INITIALIZED',
    102: 'HIPRAND_STATUS_ALLOCATION_FAILED',
    103: 'HIPRAND_STATUS_TYPE_ERROR',
    104: 'HIPRAND_STATUS_OUT_OF_RANGE',
    105: 'HIPRAND_STATUS_LENGTH_NOT_MULTIPLE',
    106: 'HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED',
    201: 'HIPRAND_STATUS_LAUNCH_FAILURE',
    202: 'HIPRAND_STATUS_PREEXISTING_FAILURE',
    203: 'HIPRAND_STATUS_INITIALIZATION_FAILED',
    204: 'HIPRAND_STATUS_ARCH_MISMATCH',
    999: 'HIPRAND_STATUS_INTERNAL_ERROR',
}


class CURANDError(RuntimeError):

    def __init__(self, status):
        self.status = status
        super(CURANDError, self).__init__(STATUS[status])

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CURANDError(status)


###############################################################################
# Generator
###############################################################################

cpdef size_t createGenerator(int rng_type) except? 0:
    cdef Generator generator
    with nogil:
        status = hiprandCreateGenerator(&generator, <RngType>rng_type)
    check_status(status)
    return <size_t>generator


cpdef destroyGenerator(size_t generator):
    status = hiprandDestroyGenerator(<Generator>generator)
    check_status(status)


cpdef int getVersion() except? -1:
    cdef int version
    status = hiprandGetVersion(&version)
    check_status(status)
    return version


cpdef setStream(size_t generator, size_t stream):
    # TODO(leofang): The support of stream capture is not mentioned at all in
    # the cuRAND docs (as of CUDA 11.5), so we disable this functionality.
    if not runtime._is_hip_environment and runtime.streamIsCapturing(stream):
        raise NotImplementedError(
            'calling cuRAND API during stream capture is currently '
            'unsupported')

    status = hiprandSetStream(<Generator>generator, <Stream>stream)
    check_status(status)


cdef _setStream(size_t generator):
    """Set current stream"""
    setStream(generator, stream_module.get_current_stream_ptr())


cpdef setPseudoRandomGeneratorSeed(size_t generator, unsigned long long seed):
    status = hiprandSetPseudoRandomGeneratorSeed(<Generator>generator, seed)
    check_status(status)


cpdef setGeneratorOffset(size_t generator, unsigned long long offset):
    status = hiprandSetGeneratorOffset(<Generator>generator, offset)
    check_status(status)


cpdef setGeneratorOrdering(size_t generator, int order):
    status = hiprandSetGeneratorOrdering(<Generator>generator, <Ordering>order)
    check_status(status)


###############################################################################
# Generation functions
###############################################################################

cpdef generate(size_t generator, size_t outputPtr, size_t num):
    _setStream(generator)
    status = hiprandGenerate(
        <Generator>generator, <unsigned int*>outputPtr, num)
    check_status(status)


cpdef generateLongLong(size_t generator, size_t outputPtr, size_t num):
    _setStream(generator)
    status = hiprandGenerateLongLong(
        <Generator>generator, <unsigned long long*>outputPtr, num)
    check_status(status)


cpdef generateUniform(size_t generator, size_t outputPtr, size_t num):
    _setStream(generator)
    status = hiprandGenerateUniform(
        <Generator>generator, <float*>outputPtr, num)
    check_status(status)


cpdef generateUniformDouble(size_t generator, size_t outputPtr, size_t num):
    _setStream(generator)
    status = hiprandGenerateUniformDouble(
        <Generator>generator, <double*>outputPtr, num)
    check_status(status)


cpdef generateNormal(size_t generator, size_t outputPtr, size_t n,
                     float mean, float stddev):
    if n % 2 == 1:
        msg = ('hiprandGenerateNormal can only generate even number of '
               'random variables simultaneously. See issue #390 for detail.')
        raise ValueError(msg)
    _setStream(generator)
    status = hiprandGenerateNormal(
        <Generator>generator, <float*>outputPtr, n, mean, stddev)
    check_status(status)


cpdef generateNormalDouble(size_t generator, size_t outputPtr, size_t n,
                           float mean, float stddev):
    if n % 2 == 1:
        msg = ('hiprandGenerateNormalDouble can only generate even number of '
               'random variables simultaneously. See issue #390 for detail.')
        raise ValueError(msg)
    _setStream(generator)
    status = hiprandGenerateNormalDouble(
        <Generator>generator, <double*>outputPtr, n, mean, stddev)
    check_status(status)


def generateLogNormal(size_t generator, size_t outputPtr, size_t n,
                      float mean, float stddev):
    if n % 2 == 1:
        msg = ('hiprandGenerateLogNormal can only generate even number of '
               'random variables simultaneously. See issue #390 for detail.')
        raise ValueError(msg)
    _setStream(generator)
    status = hiprandGenerateLogNormal(
        <Generator>generator, <float*>outputPtr, n, mean, stddev)
    check_status(status)


cpdef generateLogNormalDouble(size_t generator, size_t outputPtr, size_t n,
                              float mean, float stddev):
    if n % 2 == 1:
        msg = ('hiprandGenerateLogNormalDouble can only generate even number '
               'of random variables simultaneously. See issue #390 for '
               'detail.')
        raise ValueError(msg)
    _setStream(generator)
    status = hiprandGenerateLogNormalDouble(
        <Generator>generator, <double*>outputPtr, n, mean, stddev)
    check_status(status)


cpdef generatePoisson(size_t generator, size_t outputPtr, size_t n,
                      double lam):
    _setStream(generator)
    status = hiprandGeneratePoisson(
        <Generator>generator, <unsigned int*>outputPtr, n, lam)
    check_status(status)
