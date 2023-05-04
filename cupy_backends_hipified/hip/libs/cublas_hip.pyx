# distutils: language = c++

"""Thin wrapper of CUBLAS."""

cimport cython  # NOQA

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_complex.h':
    ctypedef struct rocblas_float_complex 'rocblas_float_complex':
        float x, y

    ctypedef struct rocblas_double_complex 'rocblas_double_complex':
        double x, y

cdef extern from '../../cupy_blas.h' nogil:
    ctypedef void* Stream 'hipStream_t'
    ctypedef int DataType 'hipDataType'

    # Context
    int rocblas_create_handle(Handle* handle)
    int rocblas_destroy_handle(Handle handle)
    int rocblas_get_version(Handle handle, int* version)
    int rocblas_get_pointer_mode(Handle handle, PointerMode* mode)
    int rocblas_set_pointer_mode(Handle handle, PointerMode mode)

    # Stream
    int rocblas_set_stream(Handle handle, Stream streamId)
    int rocblas_get_stream(Handle handle, Stream* streamId)

    # Math Mode
    int rocblas_set_math_mode(Handle handle, Math mode)
    int rocblas_get_math_mode(Handle handle, Math* mode)

    # BLAS Level 1
    int rocblas_isamax(Handle handle, int n, float* x, int incx,
                     int* result)
    int rocblas_idamax(Handle handle, int n, double* x, int incx,
                     int* result)
    int rocblas_icamax(Handle handle, int n, rocblas_float_complex* x, int incx,
                     int* result)
    int rocblas_izamax(Handle handle, int n, rocblas_double_complex* x, int incx,
                     int* result)
    int rocblas_isamin(Handle handle, int n, float* x, int incx,
                     int* result)
    int rocblas_idamin(Handle handle, int n, double* x, int incx,
                     int* result)
    int rocblas_icamin(Handle handle, int n, rocblas_float_complex* x, int incx,
                     int* result)
    int rocblas_izamin(Handle handle, int n, rocblas_double_complex* x, int incx,
                     int* result)
    int rocblas_sasum(Handle handle, int n, float* x, int incx,
                    float* result)
    int rocblas_dasum(Handle handle, int n, double* x, int incx,
                    double* result)
    int rocblas_scasum(Handle handle, int n, rocblas_float_complex* x, int incx,
                     float* result)
    int rocblas_dzasum(Handle handle, int n, rocblas_double_complex* x, int incx,
                     double* result)
    int rocblas_saxpy(Handle handle, int n, float* alpha, float* x,
                    int incx, float* y, int incy)
    int rocblas_daxpy(Handle handle, int n, double* alpha, double* x,
                    int incx, double* y, int incy)
    int rocblas_caxpy(Handle handle, int n, rocblas_float_complex* alpha, rocblas_float_complex* x,
                    int incx, rocblas_float_complex* y, int incy)
    int rocblas_zaxpy(Handle handle, int n, rocblas_double_complex* alpha,
                    rocblas_double_complex* x, int incx, rocblas_double_complex* y, int incy)
    int rocblas_sdot(Handle handle, int n, float* x, int incx,
                   float* y, int incy, float* result)
    int rocblas_ddot(Handle handle, int n, double* x, int incx,
                   double* y, int incy, double* result)
    int rocblas_cdotu(Handle handle, int n, rocblas_float_complex* x, int incx,
                    rocblas_float_complex* y, int incy, rocblas_float_complex* result)
    int rocblas_cdotc(Handle handle, int n, rocblas_float_complex* x, int incx,
                    rocblas_float_complex* y, int incy, rocblas_float_complex* result)
    int rocblas_zdotu(Handle handle, int n, rocblas_double_complex* x, int incx,
                    rocblas_double_complex* y, int incy,
                    rocblas_double_complex* result)
    int rocblas_zdotc(Handle handle, int n, rocblas_double_complex* x, int incx,
                    rocblas_double_complex* y, int incy,
                    rocblas_double_complex* result)
    int rocblas_snrm2(Handle handle, int n, float* x, int incx, float* result)
    int rocblas_dnrm2(Handle handle, int n, double* x, int incx, double* result)
    int rocblas_scnrm2(Handle handle, int n, rocblas_float_complex* x, int incx,
                     float* result)
    int rocblas_dznrm2(Handle handle, int n, rocblas_double_complex* x, int incx,
                     double* result)
    int rocblas_sscal(Handle handle, int n, float* alpha, float* x, int incx)
    int rocblas_dscal(Handle handle, int n, double* alpha, double* x, int incx)
    int rocblas_cscal(Handle handle, int n, rocblas_float_complex* alpha,
                    rocblas_float_complex* x, int incx)
    int rocblas_csscal(Handle handle, int n, float* alpha,
                     rocblas_float_complex* x, int incx)
    int rocblas_zscal(Handle handle, int n, rocblas_double_complex* alpha,
                    rocblas_double_complex* x, int incx)
    int rocblas_zdscal(Handle handle, int n, double* alpha,
                     rocblas_double_complex* x, int incx)

    # BLAS Level 2
    int rocblas_sgemv(
        Handle handle, Operation trans, int m, int n, float* alpha,
        float* A, int lda, float* x, int incx, float* beta,
        float* y, int incy)
    int rocblas_dgemv(
        Handle handle, Operation trans, int m, int n, double* alpha,
        double* A, int lda, double* x, int incx, double* beta,
        double* y, int incy)
    int rocblas_cgemv(
        Handle handle, Operation trans, int m, int n, rocblas_float_complex* alpha,
        rocblas_float_complex* A, int lda, rocblas_float_complex* x, int incx, rocblas_float_complex* beta,
        rocblas_float_complex* y, int incy)
    int rocblas_zgemv(
        Handle handle, Operation trans, int m, int n, rocblas_double_complex* alpha,
        rocblas_double_complex* A, int lda, rocblas_double_complex* x, int incx,
        rocblas_double_complex* beta, rocblas_double_complex* y, int incy)
    int rocblas_sger(
        Handle handle, int m, int n, float* alpha, float* x, int incx,
        float* y, int incy, float* A, int lda)
    int rocblas_dger(
        Handle handle, int m, int n, double* alpha, double* x,
        int incx, double* y, int incy, double* A, int lda)
    int rocblas_cgeru(
        Handle handle, int m, int n, rocblas_float_complex* alpha, rocblas_float_complex* x,
        int incx, rocblas_float_complex* y, int incy, rocblas_float_complex* A, int lda)
    int rocblas_cgerc(
        Handle handle, int m, int n, rocblas_float_complex* alpha, rocblas_float_complex* x,
        int incx, rocblas_float_complex* y, int incy, rocblas_float_complex* A, int lda)
    int rocblas_zgeru(
        Handle handle, int m, int n, rocblas_double_complex* alpha,
        rocblas_double_complex* x, int incx, rocblas_double_complex* y, int incy,
        rocblas_double_complex* A, int lda)
    int rocblas_zgerc(
        Handle handle, int m, int n, rocblas_double_complex* alpha,
        rocblas_double_complex* x, int incx, rocblas_double_complex* y, int incy,
        rocblas_double_complex* A, int lda)
    int rocblas_ssbmv(
        Handle handle, FillMode uplo, int n, int k,
        const float* alpha, const float* A, int lda,
        const float* x, int incx, const float* beta, float* y, int incy)
    int rocblas_dsbmv(
        Handle handle, FillMode uplo, int n, int k,
        const double* alpha, const double* A, int lda,
        const double* x, int incx, const double* beta, double* y, int incy)

    # BLAS Level 3
    int rocblas_sgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, float* alpha, float* A, int lda, float* B,
        int ldb, float* beta, float* C, int ldc)
    int rocblas_dgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, double* alpha, double* A, int lda, double* B,
        int ldb, double* beta, double* C, int ldc)
    int rocblas_cgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, rocblas_float_complex* alpha, rocblas_float_complex* A, int lda,
        rocblas_float_complex* B, int ldb, rocblas_float_complex* beta, rocblas_float_complex* C,
        int ldc)
    int rocblas_zgemm(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, rocblas_double_complex* alpha, rocblas_double_complex* A, int lda,
        rocblas_double_complex* B, int ldb, rocblas_double_complex* beta,
        rocblas_double_complex* C, int ldc)
    int rocblas_sgemm_batched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const float* alpha, const float** Aarray,
        int lda, const float** Barray, int ldb, const float* beta,
        float** Carray, int ldc, int batchCount)
    int rocblas_dgemm_batched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const double* alpha, const double** Aarray,
        int lda, const double** Barray, int ldb, const double* beta,
        double** Carray, int ldc, int batchCount)
    int rocblas_cgemm_batched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const rocblas_float_complex* alpha, const rocblas_float_complex** Aarray,
        int lda, const rocblas_float_complex** Barray, int ldb, const rocblas_float_complex* beta,
        rocblas_float_complex** Carray, int ldc, int batchCount)
    int rocblas_zgemm_batched(
        Handle handle, Operation transa, Operation transb, int m,
        int n, int k, const rocblas_double_complex* alpha,
        const rocblas_double_complex** Aarray, int lda,
        const rocblas_double_complex** Barray, int ldb,
        const rocblas_double_complex* beta, rocblas_double_complex** Carray, int ldc,
        int batchCount)
    int rocblas_sgemm_strided_batched(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k, const float* alpha,
        const float* A, int lda, long long strideA,
        const float* B, int ldb, long long strideB,
        const float* beta,
        float* C, int ldc, long long strideC, int batchCount)
    int rocblas_dgemm_strided_batched(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k, const double* alpha,
        const double* A, int lda, long long strideA,
        const double* B, int ldb, long long strideB,
        const double* beta,
        double* C, int ldc, long long strideC, int batchCount)
    int rocblas_cgemm_strided_batched(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k, const rocblas_float_complex* alpha,
        const rocblas_float_complex* A, int lda, long long strideA,
        const rocblas_float_complex* B, int ldb, long long strideB,
        const rocblas_float_complex* beta,
        rocblas_float_complex* C, int ldc, long long strideC, int batchCount)
    int rocblas_zgemm_strided_batched(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k, const rocblas_double_complex* alpha,
        const rocblas_double_complex* A, int lda, long long strideA,
        const rocblas_double_complex* B, int ldb, long long strideB,
        const rocblas_double_complex* beta,
        rocblas_double_complex* C, int ldc, long long strideC, int batchCount)
    int rocblas_strsm(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, int m, int n, const float* alpha,
        const float* A, int lda, float* B, int ldb)
    int rocblas_dtrsm(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, int m, int n, const double* alpha,
        const double* A, int lda, double* B, int ldb)
    int rocblas_ctrsm(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, int m, int n, const rocblas_float_complex* alpha,
        const rocblas_float_complex* A, int lda, rocblas_float_complex* B, int ldb)
    int rocblas_ztrsm(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, int m, int n, const rocblas_double_complex* alpha,
        const rocblas_double_complex* A, int lda, rocblas_double_complex* B, int ldb)
    int rocblas_ssyrk(
        Handle handle, FillMode uplo, Operation trans, int n, int k,
        float* alpha, float* A, int lda,
        float* beta, float* C, int ldc)
    int rocblas_dsyrk(
        Handle handle, FillMode uplo, Operation trans, int n, int k,
        double* alpha, double* A, int lda,
        double* beta, double* C, int ldc)
    int rocblas_csyrk(
        Handle handle, FillMode uplo, Operation trans, int n, int k,
        rocblas_float_complex* alpha, rocblas_float_complex* A, int lda,
        rocblas_float_complex* beta, rocblas_float_complex* C, int ldc)
    int rocblas_zsyrk(
        Handle handle, FillMode uplo, Operation trans, int n, int k,
        rocblas_double_complex* alpha, rocblas_double_complex* A, int lda,
        rocblas_double_complex* beta, rocblas_double_complex* C, int ldc)

    # BLAS extension
    int rocblas_sgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const float* alpha, const float* A, int lda,
        const float* beta, const float* B, int ldb,
        float* C, int ldc)
    int rocblas_dgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const double* alpha, const double* A, int lda,
        const double* beta, const double* B, int ldb,
        double* C, int ldc)
    int rocblas_cgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const rocblas_float_complex* alpha, const rocblas_float_complex* A, int lda,
        const rocblas_float_complex* beta, const rocblas_float_complex* B, int ldb,
        rocblas_float_complex* C, int ldc)
    int rocblas_zgeam(
        Handle handle, Operation transa, Operation transb, int m, int n,
        const rocblas_double_complex* alpha, const rocblas_double_complex* A, int lda,
        const rocblas_double_complex* beta, const rocblas_double_complex* B, int ldb,
        rocblas_double_complex* C, int ldc)
    int rocblas_sdgmm(
        Handle handle, SideMode mode, int m, int n, const float* A, int lda,
        const float* x, int incx, float* C, int ldc)
    int rocblas_ddgmm(
        Handle handle, SideMode mode, int m, int n, const double* A, int lda,
        const double* x, int incx, double* C, int ldc)
    int rocblas_cdgmm(
        Handle handle, SideMode mode, int m, int n, const rocblas_float_complex* A,
        int lda, const rocblas_float_complex* x, int incx, rocblas_float_complex* C, int ldc)
    int rocblas_zdgmm(
        Handle handle, SideMode mode, int m, int n, const rocblas_double_complex* A,
        int lda, const rocblas_double_complex* x, int incx, rocblas_double_complex* C,
        int ldc)
    int rocblas_sgemmex(
        Handle handle, Operation transa,
        Operation transb, int m, int n, int k,
        const float *alpha, const void *A, DataType Atype,
        int lda, const void *B, DataType Btype, int ldb,
        const float *beta, void *C, DataType Ctype, int ldc)
    int rocblas_sgetrf_batched(
        Handle handle, int n, float **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int rocblas_dgetrf_batched(
        Handle handle, int n, double **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int rocblas_cgetrf_batched(
        Handle handle, int n, rocblas_float_complex **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int rocblas_zgetrf_batched(
        Handle handle, int n, rocblas_double_complex **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)

    int rocblas_sgetrs_batched(
        Handle handle, Operation trans, int n, int nrhs,
        const float **Aarray, int lda, const int *devIpiv,
        float **Barray, int ldb, int *info, int batchSize)
    int rocblas_dgetrs_batched(
        Handle handle, Operation trans, int n, int nrhs,
        const double **Aarray, int lda, const int *devIpiv,
        double **Barray, int ldb, int *info, int batchSize)
    int rocblas_cgetrs_batched(
        Handle handle, Operation trans, int n, int nrhs,
        const rocblas_float_complex **Aarray, int lda, const int *devIpiv,
        rocblas_float_complex **Barray, int ldb, int *info, int batchSize)
    int rocblas_zgetrs_batched(
        Handle handle, Operation trans, int n, int nrhs,
        const rocblas_double_complex **Aarray, int lda, const int *devIpiv,
        rocblas_double_complex **Barray, int ldb, int *info, int batchSize)

    int rocblas_sgetri_batched(
        Handle handle, int n, const float **Aarray, int lda,
        int *PivotArray, float *Carray[], int ldc, int *infoArray,
        int batchSize)
    int rocblas_dgetri_batched(
        Handle handle, int n, const double **Aarray, int lda,
        int *PivotArray, double *Carray[], int ldc, int *infoArray,
        int batchSize)
    int rocblas_cgetri_batched(
        Handle handle, int n, const rocblas_float_complex **Aarray, int lda,
        int *PivotArray, rocblas_float_complex *Carray[], int ldc, int *infoArray,
        int batchSize)
    int rocblas_zgetri_batched(
        Handle handle, int n, const rocblas_double_complex **Aarray, int lda,
        int *PivotArray, rocblas_double_complex *Carray[], int ldc, int *infoArray,
        int batchSize)
    int rocblas_gemmex(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k,
        const void *alpha,
        const void *A, DataType Atype, int lda,
        const void *B, DataType Btype, int ldb,
        const void *beta,
        void *C, DataType Ctype, int ldc,
        DataType computetype, GemmAlgo algo)
    int cublasGemmEx_v11(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k,
        const void *alpha,
        const void *A, DataType Atype, int lda,
        const void *B, DataType Btype, int ldb,
        const void *beta,
        void *C, DataType Ctype, int ldc,
        ComputeType computetype, GemmAlgo algo)
    int cublasGemmStridedBatchedEx(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k,
        const void *alpha,
        const void *A, DataType Atype, int lda, long long strideA,
        const void *B, DataType Btype, int ldb, long long strideB,
        const void *beta,
        void *C, DataType Ctype, int ldc, long long strideC,
        int batchCount, DataType computetype, GemmAlgo algo)
    int cublasGemmStridedBatchedEx_v11(
        Handle handle, Operation transa, Operation transb,
        int m, int n, int k,
        const void *alpha,
        const void *A, DataType Atype, int lda, long long strideA,
        const void *B, DataType Btype, int ldb, long long strideB,
        const void *beta,
        void *C, DataType Ctype, int ldc, long long strideC,
        int batchCount, ComputeType computetype, GemmAlgo algo)
    int rocblas_stpttr(
        Handle handle, FillMode uplo, int n, const float *AP, float *A,
        int lda)
    int rocblas_dtpttr(
        Handle handle, FillMode uplo, int n, const double *AP, double *A,
        int lda)
    int rocblas_strttp(
        Handle handle, FillMode uplo, int n, const float *A, int lda,
        float *AP)
    int rocblas_dtrttp(
        Handle handle, FillMode uplo, int n, const double *A, int lda,
        double *AP)


###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    0: 'rocblas_status_success',
    1: 'rocblas_status_invalid_handle',
    3: 'rocblas_status_memory_error',
    7: 'rocblas_status_invalid_pointer',
    8: 'rocblas_status_not_implemented',
    11: 'rocblas_status_internal_error',
    13: 'rocblas_status_internal_error',
    14: 'rocblas_status_internal_error',
    15: 'rocblas_status_not_implemented',
    16: 'CUBLAS_STATUS_LICENSE_ERROR',
}


cdef dict HIP_STATUS = {
    0: 'HIPBLAS_STATUS_SUCCESS',
    1: 'HIPBLAS_STATUS_NOT_INITIALIZED',
    2: 'HIPBLAS_STATUS_ALLOC_FAILED',
    3: 'HIPBLAS_STATUS_INVALID_VALUE',
    4: 'HIPBLAS_STATUS_MAPPING_ERROR',
    5: 'HIPBLAS_STATUS_EXECUTION_FAILED',
    6: 'HIPBLAS_STATUS_INTERNAL_ERROR',
    7: 'HIPBLAS_STATUS_NOT_SUPPORTED',
    8: 'HIPBLAS_STATUS_ARCH_MISMATCH',
    9: 'HIPBLAS_STATUS_HANDLE_IS_NULLPTR',
}


class CUBLASError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef str err
        if runtime._is_hip_environment:
            err = HIP_STATUS[status]
        else:
            err = STATUS[status]
        super(CUBLASError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUBLASError(status)


###############################################################################
# Context
###############################################################################

cpdef intptr_t create() except? 0:
    cdef Handle handle
    with nogil:
        status = rocblas_create_handle(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    with nogil:
        status = rocblas_destroy_handle(<Handle>handle)
    check_status(status)


cpdef int getVersion(intptr_t handle) except? -1:
    cdef int version
    with nogil:
        status = rocblas_get_version(<Handle>handle, &version)
    check_status(status)
    return version


cpdef int getPointerMode(intptr_t handle) except? -1:
    cdef PointerMode mode
    with nogil:
        status = rocblas_get_pointer_mode(<Handle>handle, &mode)
    check_status(status)
    return mode


cpdef setPointerMode(intptr_t handle, int mode):
    with nogil:
        status = rocblas_set_pointer_mode(<Handle>handle, <PointerMode>mode)
    check_status(status)


###############################################################################
# Stream
###############################################################################

cpdef setStream(intptr_t handle, size_t stream):
    # TODO(leofang): It seems most of cuBLAS APIs support stream capture (as of
    # CUDA 11.5) under certain conditions, see
    # https://docs.nvidia.com/cuda/cublas/index.html#CUDA-graphs
    # Before we come up with a robust strategy to test the support conditions,
    # we disable this functionality.
    if not runtime._is_hip_environment and runtime.streamIsCapturing(stream):
        raise NotImplementedError(
            'calling cuBLAS API during stream capture is currently '
            'unsupported')

    with nogil:
        status = rocblas_set_stream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef size_t getStream(intptr_t handle) except? 0:
    cdef Stream stream
    with nogil:
        status = rocblas_get_stream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


cdef _setStream(intptr_t handle):
    """Set current stream"""
    setStream(handle, stream_module.get_current_stream_ptr())

###############################################################################
# Math Mode
###############################################################################

cpdef setMathMode(intptr_t handle, int mode):
    with nogil:
        status = rocblas_set_math_mode(<Handle>handle, <Math>mode)
    check_status(status)


cpdef int getMathMode(intptr_t handle) except? -1:
    cdef Math mode
    with nogil:
        status = rocblas_get_math_mode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


###############################################################################
# BLAS Level 1
###############################################################################

cpdef isamax(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_isamax(
            <Handle>handle, n, <float*>x, incx, <int*>result)
    check_status(status)

cpdef idamax(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_idamax(
            <Handle>handle, n, <double*>x, incx, <int*>result)
    check_status(status)

cpdef icamax(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_icamax(
            <Handle>handle, n, <rocblas_float_complex*>x, incx, <int*>result)
    check_status(status)

cpdef izamax(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_izamax(
            <Handle>handle, n, <rocblas_double_complex*>x, incx, <int*>result)
    check_status(status)


cpdef isamin(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_isamin(
            <Handle>handle, n, <float*>x, incx, <int*>result)
    check_status(status)

cpdef idamin(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_idamin(
            <Handle>handle, n, <double*>x, incx, <int*>result)
    check_status(status)

cpdef icamin(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_icamin(
            <Handle>handle, n, <rocblas_float_complex*>x, incx, <int*>result)
    check_status(status)

cpdef izamin(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_izamin(
            <Handle>handle, n, <rocblas_double_complex*>x, incx, <int*>result)
    check_status(status)


cpdef sasum(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_sasum(
            <Handle>handle, n, <float*>x, incx, <float*>result)
    check_status(status)

cpdef dasum(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_dasum(
            <Handle>handle, n, <double*>x, incx, <double*>result)
    check_status(status)

cpdef scasum(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_scasum(
            <Handle>handle, n, <rocblas_float_complex*>x, incx, <float*>result)
    check_status(status)

cpdef dzasum(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_dzasum(
            <Handle>handle, n, <rocblas_double_complex*>x, incx, <double*>result)
    check_status(status)


cpdef saxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_saxpy(
            <Handle>handle, n, <float*>alpha, <float*>x, incx, <float*>y, incy)
    check_status(status)

cpdef daxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_daxpy(
            <Handle>handle, n, <double*>alpha, <double*>x, incx, <double*>y,
            incy)
    check_status(status)

cpdef caxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_caxpy(
            <Handle>handle, n, <rocblas_float_complex*>alpha, <rocblas_float_complex*>x, incx,
            <rocblas_float_complex*>y, incy)
    check_status(status)

cpdef zaxpy(intptr_t handle, int n, size_t alpha, size_t x, int incx, size_t y,
            int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_zaxpy(
            <Handle>handle, n, <rocblas_double_complex*>alpha, <rocblas_double_complex*>x,
            incx, <rocblas_double_complex*>y, incy)
    check_status(status)


cpdef sdot(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_sdot(
            <Handle>handle, n, <float*>x, incx, <float*>y, incy,
            <float*>result)
    check_status(status)

cpdef ddot(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
           size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_ddot(
            <Handle>handle, n, <double*>x, incx, <double*>y, incy,
            <double*>result)
    check_status(status)

cpdef cdotu(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_cdotu(
            <Handle>handle, n, <rocblas_float_complex*>x, incx, <rocblas_float_complex*>y, incy,
            <rocblas_float_complex*>result)
    check_status(status)

cpdef cdotc(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_cdotc(
            <Handle>handle, n, <rocblas_float_complex*>x, incx, <rocblas_float_complex*>y, incy,
            <rocblas_float_complex*>result)
    check_status(status)

cpdef zdotu(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_zdotu(
            <Handle>handle, n, <rocblas_double_complex*>x, incx,
            <rocblas_double_complex*>y, incy, <rocblas_double_complex*>result)
    check_status(status)

cpdef zdotc(intptr_t handle, int n, size_t x, int incx, size_t y, int incy,
            size_t result):
    with nogil:
        status = rocblas_zdotc(
            <Handle>handle, n, <rocblas_double_complex*>x, incx,
            <rocblas_double_complex*>y, incy, <rocblas_double_complex*>result)
    check_status(status)


cpdef snrm2(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_snrm2(<Handle>handle, n, <float*>x, incx,
                             <float*>result)
    check_status(status)

cpdef dnrm2(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_dnrm2(<Handle>handle, n, <double*>x, incx,
                             <double*>result)
    check_status(status)

cpdef scnrm2(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_scnrm2(<Handle>handle, n, <rocblas_float_complex*>x, incx,
                              <float*>result)
    check_status(status)

cpdef dznrm2(intptr_t handle, int n, size_t x, int incx, size_t result):
    _setStream(handle)
    with nogil:
        status = rocblas_dznrm2(<Handle>handle, n, <rocblas_double_complex*>x, incx,
                              <double*>result)
    check_status(status)


cpdef sscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = rocblas_sscal(<Handle>handle, n, <float*>alpha,
                             <float*>x, incx)
    check_status(status)

cpdef dscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = rocblas_dscal(<Handle>handle, n, <double*>alpha,
                             <double*>x, incx)
    check_status(status)

cpdef cscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = rocblas_cscal(<Handle>handle, n, <rocblas_float_complex*>alpha,
                             <rocblas_float_complex*>x, incx)
    check_status(status)

cpdef csscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = rocblas_csscal(<Handle>handle, n, <float*>alpha,
                              <rocblas_float_complex*>x, incx)
    check_status(status)

cpdef zscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = rocblas_zscal(<Handle>handle, n, <rocblas_double_complex*>alpha,
                             <rocblas_double_complex*>x, incx)
    check_status(status)

cpdef zdscal(intptr_t handle, int n, size_t alpha, size_t x, int incx):
    _setStream(handle)
    with nogil:
        status = rocblas_zdscal(<Handle>handle, n, <double*>alpha,
                              <rocblas_double_complex*>x, incx)
    check_status(status)


###############################################################################
# BLAS Level 2
###############################################################################

cpdef sgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_sgemv(
            <Handle>handle, <Operation>trans, m, n, <float*>alpha,
            <float*>A, lda, <float*>x, incx, <float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_dgemv(
            <Handle>handle, <Operation>trans, m, n, <double*>alpha,
            <double*>A, lda, <double*>x, incx, <double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_cgemv(
            <Handle>handle, <Operation>trans, m, n, <rocblas_float_complex*>alpha,
            <rocblas_float_complex*>A, lda, <rocblas_float_complex*>x, incx, <rocblas_float_complex*>beta,
            <rocblas_float_complex*>y, incy)
    check_status(status)


cpdef zgemv(intptr_t handle, int trans, int m, int n, size_t alpha, size_t A,
            int lda, size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_zgemv(
            <Handle>handle, <Operation>trans, m, n, <rocblas_double_complex*>alpha,
            <rocblas_double_complex*>A, lda, <rocblas_double_complex*>x, incx,
            <rocblas_double_complex*>beta, <rocblas_double_complex*>y, incy)
    check_status(status)


cpdef sger(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = rocblas_sger(
            <Handle>handle, m, n, <float*>alpha, <float*>x, incx, <float*>y,
            incy, <float*>A, lda)
    check_status(status)


cpdef dger(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
           size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = rocblas_dger(
            <Handle>handle, m, n, <double*>alpha, <double*>x, incx, <double*>y,
            incy, <double*>A, lda)
    check_status(status)


cpdef cgeru(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = rocblas_cgeru(
            <Handle>handle, m, n, <rocblas_float_complex*>alpha, <rocblas_float_complex*>x, incx,
            <rocblas_float_complex*>y, incy, <rocblas_float_complex*>A, lda)
    check_status(status)


cpdef cgerc(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = rocblas_cgerc(
            <Handle>handle, m, n, <rocblas_float_complex*>alpha, <rocblas_float_complex*>x, incx,
            <rocblas_float_complex*>y, incy, <rocblas_float_complex*>A, lda)
    check_status(status)


cpdef zgeru(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = rocblas_zgeru(
            <Handle>handle, m, n, <rocblas_double_complex*>alpha,
            <rocblas_double_complex*>x, incx, <rocblas_double_complex*>y, incy,
            <rocblas_double_complex*>A, lda)
    check_status(status)


cpdef zgerc(intptr_t handle, int m, int n, size_t alpha, size_t x, int incx,
            size_t y, int incy, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = rocblas_zgerc(
            <Handle>handle, m, n, <rocblas_double_complex*>alpha,
            <rocblas_double_complex*>x, incx, <rocblas_double_complex*>y, incy,
            <rocblas_double_complex*>A, lda)
    check_status(status)


cpdef ssbmv(intptr_t handle, int uplo, int n, int k,
            size_t alpha, size_t A, int lda,
            size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_ssbmv(
            <Handle>handle, <FillMode>uplo, n, k,
            <float*>alpha, <float*>A, lda,
            <float*>x, incx, <float*>beta, <float*>y, incy)
    check_status(status)


cpdef dsbmv(intptr_t handle, int uplo, int n, int k,
            size_t alpha, size_t A, int lda,
            size_t x, int incx, size_t beta, size_t y, int incy):
    _setStream(handle)
    with nogil:
        status = rocblas_dsbmv(
            <Handle>handle, <FillMode>uplo, n, k,
            <double*>alpha, <double*>A, lda,
            <double*>x, incx, <double*>beta, <double*>y, incy)
    check_status(status)


###############################################################################
# BLAS Level 3
###############################################################################

cpdef sgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_sgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <float*>alpha, <float*>A, lda, <float*>B, ldb, <float*>beta,
            <float*>C, ldc)
    check_status(status)


cpdef dgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_dgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <double*>alpha, <double*>A, lda, <double*>B, ldb, <double*>beta,
            <double*>C, ldc)
    check_status(status)


cpdef cgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_cgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <rocblas_float_complex*>alpha, <rocblas_float_complex*>A, lda, <rocblas_float_complex*>B, ldb,
            <rocblas_float_complex*>beta, <rocblas_float_complex*>C, ldc)
    check_status(status)


cpdef zgemm(intptr_t handle, int transa, int transb,
            int m, int n, int k, size_t alpha, size_t A, int lda,
            size_t B, int ldb, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_zgemm(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <rocblas_double_complex*>alpha, <rocblas_double_complex*>A, lda,
            <rocblas_double_complex*>B, ldb, <rocblas_double_complex*>beta,
            <rocblas_double_complex*>C, ldc)
    check_status(status)


cpdef sgemmBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        size_t beta, size_t Carray, int ldc, int batchCount):
    _setStream(handle)
    with nogil:
        status = rocblas_sgemm_batched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <float*>alpha, <const float**>Aarray, lda, <const float**>Barray,
            ldb, <float*>beta, <float**>Carray, ldc, batchCount)
    check_status(status)


cpdef dgemmBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        size_t beta, size_t Carray, int ldc, int batchCount):
    _setStream(handle)
    with nogil:
        status = rocblas_dgemm_batched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <double*>alpha, <const double**>Aarray, lda,
            <const double**>Barray, ldb, <double*>beta,
            <double**>Carray, ldc, batchCount)
    check_status(status)


cpdef cgemmBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        size_t beta, size_t Carray, int ldc, int batchCount):
    _setStream(handle)
    with nogil:
        status = rocblas_cgemm_batched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <rocblas_float_complex*>alpha, <const rocblas_float_complex**>Aarray, lda,
            <const rocblas_float_complex**>Barray, ldb, <rocblas_float_complex*>beta,
            <rocblas_float_complex**>Carray, ldc, batchCount)
    check_status(status)


cpdef zgemmBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t Aarray, int lda, size_t Barray, int ldb,
        size_t beta, size_t Carray, int ldc, int batchCount):
    _setStream(handle)
    with nogil:
        status = rocblas_zgemm_batched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <rocblas_double_complex*>alpha, <const rocblas_double_complex**>Aarray, lda,
            <const rocblas_double_complex**>Barray, ldb, <rocblas_double_complex*>beta,
            <rocblas_double_complex**>Carray, ldc, batchCount)


cpdef sgemmStridedBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int lda, long long strideA, size_t B, int ldb,
        long long strideB, size_t beta, size_t C, int ldc, long long strideC,
        int batchCount):
    _setStream(handle)
    with nogil:
        status = rocblas_sgemm_strided_batched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const float*>alpha,
            <const float*>A, lda, <long long>strideA,
            <const float*>B, ldb, <long long>strideB,
            <const float*>beta,
            <float*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef dgemmStridedBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int lda, long long strideA, size_t B, int ldb,
        long long strideB, size_t beta, size_t C, int ldc, long long strideC,
        int batchCount):
    _setStream(handle)
    with nogil:
        status = rocblas_dgemm_strided_batched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const double*>alpha,
            <const double*>A, lda, <long long>strideA,
            <const double*>B, ldb, <long long>strideB,
            <const double*>beta,
            <double*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef cgemmStridedBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int lda, long long strideA, size_t B, int ldb,
        long long strideB, size_t beta, size_t C, int ldc, long long strideC,
        int batchCount):
    _setStream(handle)
    with nogil:
        status = rocblas_cgemm_strided_batched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const rocblas_float_complex*>alpha,
            <const rocblas_float_complex*>A, lda, <long long>strideA,
            <const rocblas_float_complex*>B, ldb, <long long>strideB,
            <const rocblas_float_complex*>beta,
            <rocblas_float_complex*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef zgemmStridedBatched(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int lda, long long strideA, size_t B, int ldb,
        long long strideB, size_t beta, size_t C, int ldc, long long strideC,
        int batchCount):
    _setStream(handle)
    with nogil:
        status = rocblas_zgemm_strided_batched(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const rocblas_double_complex*>alpha,
            <const rocblas_double_complex*>A, lda, <long long>strideA,
            <const rocblas_double_complex*>B, ldb, <long long>strideB,
            <const rocblas_double_complex*>beta,
            <rocblas_double_complex*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef strsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        int m, int n, size_t alpha, size_t Aarray, int lda,
        size_t Barray, int ldb):
    _setStream(handle)
    with nogil:
        status = rocblas_strsm(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const float*>alpha, <const float*>Aarray,
            lda, <float*>Barray, ldb)
    check_status(status)


cpdef dtrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        int m, int n, size_t alpha, size_t Aarray, int lda,
        size_t Barray, int ldb):
    _setStream(handle)
    with nogil:
        status = rocblas_dtrsm(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const double*>alpha, <const double*>Aarray,
            lda, <double*>Barray, ldb)
    check_status(status)


cpdef ctrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        int m, int n, size_t alpha, size_t Aarray, int lda,
        size_t Barray, int ldb):
    _setStream(handle)
    with nogil:
        status = rocblas_ctrsm(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const rocblas_float_complex*>alpha,
            <const rocblas_float_complex*>Aarray, lda, <rocblas_float_complex*>Barray, ldb)
    check_status(status)


cpdef ztrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        int m, int n, size_t alpha, size_t Aarray, int lda,
        size_t Barray, int ldb):
    _setStream(handle)
    with nogil:
        status = rocblas_ztrsm(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const rocblas_double_complex*>alpha,
            <const rocblas_double_complex*>Aarray, lda, <rocblas_double_complex*>Barray, ldb)
    check_status(status)


cpdef ssyrk(intptr_t handle, int uplo, int trans, int n, int k,
            size_t alpha, size_t A, int lda, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_ssyrk(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const float*>alpha, <const float*>A, lda,
            <const float*>beta, <float*>C, ldc)
    check_status(status)


cpdef dsyrk(intptr_t handle, int uplo, int trans, int n, int k,
            size_t alpha, size_t A, int lda, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_dsyrk(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const double*>alpha, <const double*>A, lda,
            <const double*>beta, <double*>C, ldc)
    check_status(status)


cpdef csyrk(intptr_t handle, int uplo, int trans, int n, int k,
            size_t alpha, size_t A, int lda, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_csyrk(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const rocblas_float_complex*>alpha, <const rocblas_float_complex*>A, lda,
            <const rocblas_float_complex*>beta, <rocblas_float_complex*>C, ldc)
    check_status(status)


cpdef zsyrk(intptr_t handle, int uplo, int trans, int n, int k,
            size_t alpha, size_t A, int lda, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_zsyrk(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const rocblas_double_complex*>alpha, <const rocblas_double_complex*>A, lda,
            <const rocblas_double_complex*>beta, <rocblas_double_complex*>C, ldc)
    check_status(status)


###############################################################################
# BLAS extension
###############################################################################

cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_sgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const float*>alpha, <const float*>A, lda, <const float*>beta,
            <const float*>B, ldb, <float*>C, ldc)
    check_status(status)

cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_dgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const double*>alpha, <const double*>A, lda, <const double*>beta,
            <const double*>B, ldb, <double*>C, ldc)
    check_status(status)

cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_cgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const rocblas_float_complex*>alpha, <const rocblas_float_complex*>A, lda,
            <const rocblas_float_complex*>beta, <const rocblas_float_complex*>B, ldb,
            <rocblas_float_complex*>C, ldc)
    check_status(status)

cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n,
            size_t alpha, size_t A, int lda, size_t beta, size_t B, int ldb,
            size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_zgeam(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const rocblas_double_complex*>alpha, <const rocblas_double_complex*>A, lda,
            <const rocblas_double_complex*>beta, <const rocblas_double_complex*>B, ldb,
            <rocblas_double_complex*>C, ldc)
    check_status(status)


cpdef sdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_sdgmm(
            <Handle>handle, <SideMode>mode, m, n, <const float*>A, lda,
            <const float*>x, incx, <float*>C, ldc)
    check_status(status)

cpdef ddgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_ddgmm(
            <Handle>handle, <SideMode>mode, m, n, <const double*>A, lda,
            <const double*>x, incx, <double*>C, ldc)
    check_status(status)

cpdef cdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_cdgmm(
            <Handle>handle, <SideMode>mode, m, n, <const rocblas_float_complex*>A, lda,
            <const rocblas_float_complex*>x, incx, <rocblas_float_complex*>C, ldc)
    check_status(status)

cpdef zdgmm(intptr_t handle, int mode, int m, int n, size_t A, int lda,
            size_t x, int incx, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_zdgmm(
            <Handle>handle, <SideMode>mode, m, n, <const rocblas_double_complex*>A,
            lda, <const rocblas_double_complex*>x, incx, <rocblas_double_complex*>C, ldc)
    check_status(status)


cpdef sgemmEx(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int Atype, int lda, size_t B,
        int Btype, int ldb, size_t beta, size_t C, int Ctype,
        int ldc):
    _setStream(handle)
    with nogil:
        status = rocblas_sgemmex(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const float*>alpha, <const void*>A, <DataType>Atype, lda,
            <const void*>B, <DataType>Btype, ldb, <const float*>beta,
            <void*>C, <DataType>Ctype, ldc)
    check_status(status)


cpdef sgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_sgetrf_batched(
            <Handle>handle, n, <float**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef dgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_dgetrf_batched(
            <Handle>handle, n, <double**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef cgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_cgetrf_batched(
            <Handle>handle, n, <rocblas_float_complex**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef zgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_zgetrf_batched(
            <Handle>handle, n, <rocblas_double_complex**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef int sgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_sgetrs_batched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const float**>Aarray, lda, <const int*>devIpiv,
            <float**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int dgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_dgetrs_batched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const double**>Aarray, lda, <const int*>devIpiv,
            <double**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int cgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_cgetrs_batched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const rocblas_float_complex**>Aarray, lda, <const int*>devIpiv,
            <rocblas_float_complex**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int zgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_zgetrs_batched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const rocblas_double_complex**>Aarray, lda, <const int*>devIpiv,
            <rocblas_double_complex**>Barray, ldb, <int*>info, batchSize)
    check_status(status)


cpdef sgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_sgetri_batched(
            <Handle>handle, n, <const float**>Aarray, lda, <int*>PivotArray,
            <float**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef dgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_dgetri_batched(
            <Handle>handle, n, <const double**>Aarray, lda, <int*>PivotArray,
            <double**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef cgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_cgetri_batched(
            <Handle>handle, n, <const rocblas_float_complex**>Aarray, lda,
            <int*>PivotArray,
            <rocblas_float_complex**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef zgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = rocblas_zgetri_batched(
            <Handle>handle, n, <const rocblas_double_complex**>Aarray, lda,
            <int*>PivotArray,
            <rocblas_double_complex**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef gemmEx(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha, size_t A, int Atype, int lda, size_t B,
        int Btype, int ldb, size_t beta, size_t C, int Ctype,
        int ldc, int computeType, int algo):
    _setStream(handle)
    with nogil:
        if computeType >= CUBLAS_COMPUTE_16F:
            status = cublasGemmEx_v11(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <DataType>Atype, lda,
                <const void*>B, <DataType>Btype, ldb,
                <const void*>beta,
                <void*>C, <DataType>Ctype, ldc,
                <ComputeType>computeType, <GemmAlgo>algo)
        else:
            status = rocblas_gemmex(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <DataType>Atype, lda,
                <const void*>B, <DataType>Btype, ldb,
                <const void*>beta,
                <void*>C, <DataType>Ctype, ldc,
                <DataType>computeType, <GemmAlgo>algo)
    check_status(status)


cpdef gemmStridedBatchedEx(
        intptr_t handle, int transa, int transb, int m, int n, int k,
        size_t alpha,
        size_t A, int Atype, int lda, long long strideA,
        size_t B, int Btype, int ldb, long long strideB,
        size_t beta,
        size_t C, int Ctype, int ldc, long long strideC,
        int batchCount, int computeType, int algo):
    _setStream(handle)
    with nogil:
        if computeType >= CUBLAS_COMPUTE_16F:
            status = cublasGemmStridedBatchedEx_v11(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <DataType>Atype, lda, <long long>strideA,
                <const void*>B, <DataType>Btype, ldb, <long long>strideB,
                <const void*>beta,
                <void*>C, <DataType>Ctype, ldc, <long long>strideC,
                batchCount, <ComputeType>computeType, <GemmAlgo>algo)
        else:
            status = cublasGemmStridedBatchedEx(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <DataType>Atype, lda, <long long>strideA,
                <const void*>B, <DataType>Btype, ldb, <long long>strideB,
                <const void*>beta,
                <void*>C, <DataType>Ctype, ldc, <long long>strideC,
                batchCount, <DataType>computeType, <GemmAlgo>algo)
    check_status(status)


cpdef stpttr(intptr_t handle, int uplo, int n, size_t AP, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = rocblas_stpttr(<Handle>handle, <FillMode>uplo, n,
                              <const float*>AP, <float*>A, lda)
    check_status(status)


cpdef dtpttr(intptr_t handle, int uplo, int n, size_t AP, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = rocblas_dtpttr(<Handle>handle, <FillMode>uplo, n,
                              <const double*>AP, <double*>A, lda)
    check_status(status)


cpdef strttp(intptr_t handle, int uplo, int n, size_t A, int lda, size_t AP):
    _setStream(handle)
    with nogil:
        status = rocblas_strttp(<Handle>handle, <FillMode>uplo, n,
                              <const float*>A, lda, <float*>AP)
    check_status(status)


cpdef dtrttp(intptr_t handle, int uplo, int n, size_t A, int lda, size_t AP):
    _setStream(handle)
    with nogil:
        status = rocblas_dtrttp(<Handle>handle, <FillMode>uplo, n,
                              <const double*>A, lda, <double*>AP)
    check_status(status)
