# distutils: language = c++

"""Thin wrapper of CUSOLVER."""

cimport cython  # NOQA

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module


cpdef _get_cuda_build_version():
    if CUPY_CUDA_VERSION > 0:
        return CUPY_CUDA_VERSION
    elif CUPY_HIP_VERSION > 0:
        return CUPY_HIP_VERSION
    else:
        return 0


###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_complex.h':
    ctypedef struct rocblas_float_complex 'rocblas_float_complex':
        float x, y

    ctypedef struct rocblas_double_complex 'rocblas_double_complex':
        double x, y

cdef extern from '../../cupy_lapack.h' nogil:
    ctypedef void* Stream 'hipStream_t'

    # Context
    int hipsolverDnCreate(Handle* handle)
    int cusolverSpCreate(SpHandle* handle)
    int hipsolverDnDestroy(Handle handle)
    int cusolverSpDestroy(SpHandle handle)

    # Stream
    int cusolverDnGetStream(Handle handle, Stream* streamId)
    int cusolverSpGetStream(SpHandle handle, Stream* streamId)
    int hipsolverDnSetStream(Handle handle, Stream streamId)
    int cusolverSpSetStream(SpHandle handle, Stream streamId)

    # Params
    int cusolverDnCreateParams(Params* params)
    int cusolverDnDestroyParams(Params params)

    # Library Property
    int cusolverGetProperty(LibraryPropertyType type, int* value)

    # hipLibraryPropertyType_t
    int hipLibraryMajorVersion
    int hipLibraryMinorVersion
    int hipLibraryPatchVersion

    ###########################################################################
    # Dense LAPACK Functions (Linear Solver)
    ###########################################################################

    # Cholesky factorization
    int hipsolverDnSpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    float* A, int lda, int* lwork)
    int hipsolverDnDpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    double* A, int lda, int* lwork)
    int hipsolverDnCpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    rocblas_float_complex* A, int lda, int* lwork)
    int hipsolverDnZpotrf_bufferSize(Handle handle, FillMode uplo, int n,
                                    rocblas_double_complex* A, int lda, int* lwork)

    int hipsolverDnSpotrf(Handle handle, FillMode uplo, int n,
                         float* A, int lda,
                         float* work, int lwork, int* devInfo)
    int hipsolverDnDpotrf(Handle handle, FillMode uplo, int n,
                         double* A, int lda,
                         double* work, int lwork, int* devInfo)
    int hipsolverDnCpotrf(Handle handle, FillMode uplo, int n,
                         rocblas_float_complex* A, int lda,
                         rocblas_float_complex* work, int lwork, int* devInfo)
    int hipsolverDnZpotrf(Handle handle, FillMode uplo, int n,
                         rocblas_double_complex* A, int lda,
                         rocblas_double_complex* work, int lwork, int* devInfo)

    int hipsolverDnSpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const float* A, int lda,
                         float* B, int ldb, int* devInfo)
    int hipsolverDnDpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const double* A, int lda,
                         double* B, int ldb, int* devInfo)
    int hipsolverDnCpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const rocblas_float_complex* A, int lda,
                         rocblas_float_complex* B, int ldb, int* devInfo)
    int hipsolverDnZpotrs(Handle handle, FillMode uplo, int n, int nrhs,
                         const rocblas_double_complex* A, int lda,
                         rocblas_double_complex* B, int ldb, int* devInfo)

    int hipsolverDnSpotrfBatched(Handle handle, FillMode uplo, int n,
                                float** Aarray, int lda,
                                int* infoArray, int batchSize)
    int hipsolverDnDpotrfBatched(Handle handle, FillMode uplo, int n,
                                double** Aarray, int lda,
                                int* infoArray, int batchSize)
    int hipsolverDnCpotrfBatched(Handle handle, FillMode uplo, int n,
                                rocblas_float_complex** Aarray, int lda,
                                int* infoArray, int batchSize)
    int hipsolverDnZpotrfBatched(Handle handle, FillMode uplo, int n,
                                rocblas_double_complex** Aarray, int lda,
                                int* infoArray, int batchSize)

    int hipsolverDnSpotrsBatched(Handle handle, FillMode uplo, int n,
                                int nrhs, float** Aarray, int lda,
                                float** Barray, int ldb,
                                int* devInfo, int batchSize)
    int hipsolverDnDpotrsBatched(Handle handle, FillMode uplo, int n,
                                int nrhs, double** Aarray, int lda,
                                double** Barray, int ldb,
                                int* devInfo, int batchSize)
    int hipsolverDnCpotrsBatched(Handle handle, FillMode uplo, int n,
                                int nrhs, rocblas_float_complex** Aarray, int lda,
                                rocblas_float_complex** Barray, int ldb,
                                int* devInfo, int batchSize)
    int hipsolverDnZpotrsBatched(Handle handle, FillMode uplo, int n,
                                int nrhs, rocblas_double_complex** Aarray, int lda,
                                rocblas_double_complex** Barray, int ldb,
                                int* devInfo, int batchSize)

    # LU factorization
    int hipsolverDnSgetrf_bufferSize(Handle handle, int m, int n,
                                    float* A, int lda, int* lwork)
    int hipsolverDnDgetrf_bufferSize(Handle handle, int m, int n,
                                    double* A, int lda, int* lwork)
    int hipsolverDnCgetrf_bufferSize(Handle handle, int m, int n,
                                    rocblas_float_complex* A, int lda, int* lwork)
    int hipsolverDnZgetrf_bufferSize(Handle handle, int m, int n,
                                    rocblas_double_complex* A, int lda, int* lwork)

    int hipsolverDnSgetrf(Handle handle, int m, int n,
                         float* A, int lda,
                         float* work, int* devIpiv, int* devInfo)
    int hipsolverDnDgetrf(Handle handle, int m, int n,
                         double* A, int lda,
                         double* work, int* devIpiv, int* devInfo)
    int hipsolverDnCgetrf(Handle handle, int m, int n,
                         rocblas_float_complex* A, int lda,
                         rocblas_float_complex* work, int* devIpiv, int* devInfo)
    int hipsolverDnZgetrf(Handle handle, int m, int n,
                         rocblas_double_complex* A, int lda,
                         rocblas_double_complex* work, int* devIpiv, int* devInfo)

    # TODO(anaruse): laswp

    # LU solve
    int hipsolverDnSgetrs(Handle handle, Operation trans, int n, int nrhs,
                         const float* A, int lda, const int* devIpiv,
                         float* B, int ldb, int* devInfo)
    int hipsolverDnDgetrs(Handle handle, Operation trans, int n, int nrhs,
                         const double* A, int lda, const int* devIpiv,
                         double* B, int ldb, int* devInfo)
    int hipsolverDnCgetrs(Handle handle, Operation trans, int n, int nrhs,
                         const rocblas_float_complex* A, int lda, const int* devIpiv,
                         rocblas_float_complex* B, int ldb, int* devInfo)
    int hipsolverDnZgetrs(Handle handle, Operation trans, int n, int nrhs,
                         const rocblas_double_complex* A, int lda, const int* devIpiv,
                         rocblas_double_complex* B, int ldb, int* devInfo)

    # QR factorization
    int hipsolverDnSgeqrf_bufferSize(Handle handle, int m, int n,
                                    float* A, int lda, int* lwork)
    int hipsolverDnDgeqrf_bufferSize(Handle handle, int m, int n,
                                    double* A, int lda, int* lwork)
    int hipsolverDnCgeqrf_bufferSize(Handle handle, int m, int n,
                                    rocblas_float_complex* A, int lda, int* lwork)
    int hipsolverDnZgeqrf_bufferSize(Handle handle, int m, int n,
                                    rocblas_double_complex* A, int lda, int* lwork)

    int hipsolverDnSgeqrf(Handle handle, int m, int n,
                         float* A, int lda, float* tau,
                         float* work, int lwork, int* devInfo)
    int hipsolverDnDgeqrf(Handle handle, int m, int n,
                         double* A, int lda, double* tau,
                         double* work, int lwork, int* devInfo)
    int hipsolverDnCgeqrf(Handle handle, int m, int n,
                         rocblas_float_complex* A, int lda, rocblas_float_complex* tau,
                         rocblas_float_complex* work, int lwork, int* devInfo)
    int hipsolverDnZgeqrf(Handle handle, int m, int n,
                         rocblas_double_complex* A, int lda, rocblas_double_complex* tau,
                         rocblas_double_complex* work, int lwork, int* devInfo)

    # Generate unitary matrix Q from QR factorization.
    int hipsolverDnSorgqr_bufferSize(Handle handle, int m, int n, int k,
                                    const float* A, int lda,
                                    const float* tau, int* lwork)
    int hipsolverDnDorgqr_bufferSize(Handle handle, int m, int n, int k,
                                    const double* A, int lda,
                                    const double* tau, int* lwork)
    int hipsolverDnCungqr_bufferSize(Handle handle, int m, int n, int k,
                                    const rocblas_float_complex* A, int lda,
                                    const rocblas_float_complex* tau, int* lwork)
    int hipsolverDnZungqr_bufferSize(Handle handle, int m, int n, int k,
                                    const rocblas_double_complex* A, int lda,
                                    const rocblas_double_complex* tau, int* lwork)

    int hipsolverDnSorgqr(Handle handle, int m, int n, int k,
                         float* A, int lda,
                         const float* tau,
                         float* work, int lwork, int* devInfo)
    int hipsolverDnDorgqr(Handle handle, int m, int n, int k,
                         double* A, int lda,
                         const double* tau,
                         double* work, int lwork, int* devInfo)
    int hipsolverDnCungqr(Handle handle, int m, int n, int k,
                         rocblas_float_complex* A, int lda,
                         const rocblas_float_complex* tau,
                         rocblas_float_complex* work, int lwork, int* devInfo)
    int hipsolverDnZungqr(Handle handle, int m, int n, int k,
                         rocblas_double_complex* A, int lda,
                         const rocblas_double_complex* tau,
                         rocblas_double_complex* work, int lwork, int* devInfo)

    # Compute Q**T*b in solve min||A*x = b||
    int hipsolverDnSormqr_bufferSize(Handle handle, SideMode side,
                                    Operation trans, int m, int n, int k,
                                    const float* A, int lda,
                                    const float* tau,
                                    const float* C, int ldc,
                                    int* lwork)
    int hipsolverDnDormqr_bufferSize(Handle handle, SideMode side,
                                    Operation trans, int m, int n, int k,
                                    const double* A, int lda,
                                    const double* tau,
                                    const double* C, int ldc,
                                    int* lwork)
    int hipsolverDnCunmqr_bufferSize(Handle handle, SideMode side,
                                    Operation trans, int m, int n, int k,
                                    const rocblas_float_complex* A, int lda,
                                    const rocblas_float_complex* tau,
                                    const rocblas_float_complex* C, int ldc,
                                    int* lwork)
    int hipsolverDnZunmqr_bufferSize(Handle handle, SideMode side,
                                    Operation trans, int m, int n, int k,
                                    const rocblas_double_complex* A, int lda,
                                    const rocblas_double_complex* tau,
                                    const rocblas_double_complex* C, int ldc,
                                    int* lwork)

    int hipsolverDnSormqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k,
                         const float* A, int lda,
                         const float* tau,
                         float* C, int ldc, float* work,
                         int lwork, int* devInfo)
    int hipsolverDnDormqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k,
                         const double* A, int lda,
                         const double* tau,
                         double* C, int ldc, double* work,
                         int lwork, int* devInfo)
    int hipsolverDnCunmqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k,
                         const rocblas_float_complex* A, int lda,
                         const rocblas_float_complex* tau,
                         rocblas_float_complex* C, int ldc, rocblas_float_complex* work,
                         int lwork, int* devInfo)
    int hipsolverDnZunmqr(Handle handle, SideMode side, Operation trans,
                         int m, int n, int k,
                         const rocblas_double_complex* A, int lda,
                         const rocblas_double_complex* tau,
                         rocblas_double_complex* C, int ldc, rocblas_double_complex* work,
                         int lwork, int* devInfo)

    # L*D*L**T,U*D*U**T factorization
    int hipsolverDnSsytrf_bufferSize(Handle handle, int n,
                                    float* A, int lda, int* lwork)
    int hipsolverDnDsytrf_bufferSize(Handle handle, int n,
                                    double* A, int lda, int* lwork)
    int hipsolverDnCsytrf_bufferSize(Handle handle, int n,
                                    rocblas_float_complex* A, int lda, int* lwork)
    int hipsolverDnZsytrf_bufferSize(Handle handle, int n,
                                    rocblas_double_complex* A, int lda, int* lwork)

    int hipsolverDnSsytrf(Handle handle, FillMode uplo, int n,
                         float* A, int lda, int* ipiv,
                         float* work, int lwork, int* devInfo)
    int hipsolverDnDsytrf(Handle handle, FillMode uplo, int n,
                         double* A, int lda, int* ipiv,
                         double* work, int lwork, int* devInfo)
    int hipsolverDnCsytrf(Handle handle, FillMode uplo, int n,
                         rocblas_float_complex* A, int lda, int* ipiv,
                         rocblas_float_complex* work, int lwork, int* devInfo)
    int hipsolverDnZsytrf(Handle handle, FillMode uplo, int n,
                         rocblas_double_complex* A, int lda, int* ipiv,
                         rocblas_double_complex* work, int lwork, int* devInfo)

    # Solve A * X = B using iterative refinement
    int cusolverDnZZgesv_bufferSize(Handle handle, int n, int nrhs,
                                    rocblas_double_complex *dA, int ldda, int *dipiv,
                                    rocblas_double_complex *dB, int lddb,
                                    rocblas_double_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZCgesv_bufferSize(Handle handle, int n, int nrhs,
                                    rocblas_double_complex *dA, int ldda, int *dipiv,
                                    rocblas_double_complex *dB, int lddb,
                                    rocblas_double_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZYgesv_bufferSize(Handle handle, int n, int nrhs,
                                    rocblas_double_complex *dA, int ldda, int *dipiv,
                                    rocblas_double_complex *dB, int lddb,
                                    rocblas_double_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZKgesv_bufferSize(Handle handle, int n, int nrhs,
                                    rocblas_double_complex *dA, int ldda, int *dipiv,
                                    rocblas_double_complex *dB, int lddb,
                                    rocblas_double_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCCgesv_bufferSize(Handle handle, int n, int nrhs,
                                    rocblas_float_complex *dA, int ldda, int *dipiv,
                                    rocblas_float_complex *dB, int lddb,
                                    rocblas_float_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCYgesv_bufferSize(Handle handle, int n, int nrhs,
                                    rocblas_float_complex *dA, int ldda, int *dipiv,
                                    rocblas_float_complex *dB, int lddb,
                                    rocblas_float_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCKgesv_bufferSize(Handle handle, int n, int nrhs,
                                    rocblas_float_complex *dA, int ldda, int *dipiv,
                                    rocblas_float_complex *dB, int lddb,
                                    rocblas_float_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDDgesv_bufferSize(Handle handle, int n, int nrhs,
                                    double *dA, int ldda, int *dipiv,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDSgesv_bufferSize(Handle handle, int n, int nrhs,
                                    double *dA, int ldda, int *dipiv,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDXgesv_bufferSize(Handle handle, int n, int nrhs,
                                    double *dA, int ldda, int *dipiv,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDHgesv_bufferSize(Handle handle, int n, int nrhs,
                                    double *dA, int ldda, int *dipiv,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSSgesv_bufferSize(Handle handle, int n, int nrhs,
                                    float *dA, int ldda, int *dipiv,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSXgesv_bufferSize(Handle handle, int n, int nrhs,
                                    float *dA, int ldda, int *dipiv,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSHgesv_bufferSize(Handle handle, int n, int nrhs,
                                    float *dA, int ldda, int *dipiv,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)

    int cusolverDnZZgesv(Handle handle, int n, int nrhs,
                         rocblas_double_complex *dA, int ldda, int *dipiv,
                         rocblas_double_complex *dB, int lddb,
                         rocblas_double_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZCgesv(Handle handle, int n, int nrhs,
                         rocblas_double_complex *dA, int ldda, int *dipiv,
                         rocblas_double_complex *dB, int lddb,
                         rocblas_double_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZYgesv(Handle handle, int n, int nrhs,
                         rocblas_double_complex *dA, int ldda, int *dipiv,
                         rocblas_double_complex *dB, int lddb,
                         rocblas_double_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZKgesv(Handle handle, int n, int nrhs,
                         rocblas_double_complex *dA, int ldda, int *dipiv,
                         rocblas_double_complex *dB, int lddb,
                         rocblas_double_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCCgesv(Handle handle, int n, int nrhs,
                         rocblas_float_complex *dA, int ldda, int *dipiv,
                         rocblas_float_complex *dB, int lddb,
                         rocblas_float_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCYgesv(Handle handle, int n, int nrhs,
                         rocblas_float_complex *dA, int ldda, int *dipiv,
                         rocblas_float_complex *dB, int lddb,
                         rocblas_float_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCKgesv(Handle handle, int n, int nrhs,
                         rocblas_float_complex *dA, int ldda, int *dipiv,
                         rocblas_float_complex *dB, int lddb,
                         rocblas_float_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDDgesv(Handle handle, int n, int nrhs,
                         double *dA, int ldda, int *dipiv,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDSgesv(Handle handle, int n, int nrhs,
                         double *dA, int ldda, int *dipiv,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDXgesv(Handle handle, int n, int nrhs,
                         double *dA, int ldda, int *dipiv,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDHgesv(Handle handle, int n, int nrhs,
                         double *dA, int ldda, int *dipiv,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSSgesv(Handle handle, int n, int nrhs,
                         float *dA, int ldda, int *dipiv,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSXgesv(Handle handle, int n, int nrhs,
                         float *dA, int ldda, int *dipiv,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSHgesv(Handle handle, int n, int nrhs,
                         float *dA, int ldda, int *dipiv,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)

    # Compute least square solution to A * X = B using iterative refinement
    int cusolverDnZZgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    rocblas_double_complex *dA, int ldda,
                                    rocblas_double_complex *dB, int lddb,
                                    rocblas_double_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZCgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    rocblas_double_complex *dA, int ldda,
                                    rocblas_double_complex *dB, int lddb,
                                    rocblas_double_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZYgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    rocblas_double_complex *dA, int ldda,
                                    rocblas_double_complex *dB, int lddb,
                                    rocblas_double_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnZKgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    rocblas_double_complex *dA, int ldda,
                                    rocblas_double_complex *dB, int lddb,
                                    rocblas_double_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCCgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    rocblas_float_complex *dA, int ldda,
                                    rocblas_float_complex *dB, int lddb,
                                    rocblas_float_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCYgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    rocblas_float_complex *dA, int ldda,
                                    rocblas_float_complex *dB, int lddb,
                                    rocblas_float_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnCKgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    rocblas_float_complex *dA, int ldda,
                                    rocblas_float_complex *dB, int lddb,
                                    rocblas_float_complex *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDDgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    double *dA, int ldda,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDSgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    double *dA, int ldda,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDXgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    double *dA, int ldda,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnDHgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    double *dA, int ldda,
                                    double *dB, int lddb,
                                    double *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSSgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    float *dA, int ldda,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSXgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    float *dA, int ldda,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)
    int cusolverDnSHgels_bufferSize(Handle handle, int m, int n, int nrhs,
                                    float *dA, int ldda,
                                    float *dB, int lddb,
                                    float *dX, int lddx,
                                    void *dWorkspace, size_t *lwork_bytes)

    int cusolverDnZZgels(Handle handle, int m, int n, int nrhs,
                         rocblas_double_complex *dA, int ldda,
                         rocblas_double_complex *dB, int lddb,
                         rocblas_double_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZCgels(Handle handle, int m, int n, int nrhs,
                         rocblas_double_complex *dA, int ldda,
                         rocblas_double_complex *dB, int lddb,
                         rocblas_double_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZYgels(Handle handle, int m, int n, int nrhs,
                         rocblas_double_complex *dA, int ldda,
                         rocblas_double_complex *dB, int lddb,
                         rocblas_double_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnZKgels(Handle handle, int m, int n, int nrhs,
                         rocblas_double_complex *dA, int ldda,
                         rocblas_double_complex *dB, int lddb,
                         rocblas_double_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCCgels(Handle handle, int m, int n, int nrhs,
                         rocblas_float_complex *dA, int ldda,
                         rocblas_float_complex *dB, int lddb,
                         rocblas_float_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCYgels(Handle handle, int m, int n, int nrhs,
                         rocblas_float_complex *dA, int ldda,
                         rocblas_float_complex *dB, int lddb,
                         rocblas_float_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnCKgels(Handle handle, int m, int n, int nrhs,
                         rocblas_float_complex *dA, int ldda,
                         rocblas_float_complex *dB, int lddb,
                         rocblas_float_complex *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDDgels(Handle handle, int m, int n, int nrhs,
                         double *dA, int ldda,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDSgels(Handle handle, int m, int n, int nrhs,
                         double *dA, int ldda,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDXgels(Handle handle, int m, int n, int nrhs,
                         double *dA, int ldda,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnDHgels(Handle handle, int m, int n, int nrhs,
                         double *dA, int ldda,
                         double *dB, int lddb,
                         double *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSSgels(Handle handle, int m, int n, int nrhs,
                         float *dA, int ldda,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSXgels(Handle handle, int m, int n, int nrhs,
                         float *dA, int ldda,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)
    int cusolverDnSHgels(Handle handle, int m, int n, int nrhs,
                         float *dA, int ldda,
                         float *dB, int lddb,
                         float *dX, int lddx,
                         void *dWorkspace, size_t lwork_bytes,
                         int *iter, int *dInfo)

    ###########################################################################
    # Dense LAPACK Functions (Eigenvalue Solver)
    ###########################################################################

    # Bidiagonal factorization
    int cusolverDnSgebrd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnDgebrd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnCgebrd_bufferSize(Handle handle, int m, int n, int* lwork)
    int cusolverDnZgebrd_bufferSize(Handle handle, int m, int n, int* lwork)

    int cusolverDnSgebrd(Handle handle, int m, int n,
                         float* A, int lda,
                         float* D, float* E,
                         float* tauQ, float* tauP,
                         float* Work, int lwork, int* devInfo)
    int cusolverDnDgebrd(Handle handle, int m, int n,
                         double* A, int lda,
                         double* D, double* E,
                         double* tauQ, double* tauP,
                         double* Work, int lwork, int* devInfo)
    int cusolverDnCgebrd(Handle handle, int m, int n,
                         rocblas_float_complex* A, int lda,
                         float* D, float* E,
                         rocblas_float_complex* tauQ, rocblas_float_complex* tauP,
                         rocblas_float_complex* Work, int lwork, int* devInfo)
    int cusolverDnZgebrd(Handle handle, int m, int n,
                         rocblas_double_complex* A, int lda,
                         double* D, double* E,
                         rocblas_double_complex* tauQ, rocblas_double_complex* tauP,
                         rocblas_double_complex* Work, int lwork, int* devInfo)

    # Singular value decomposition, A = U * Sigma * V^H
    int hipsolverDnSgesvd_bufferSize(Handle handle, int m, int n, int* lwork)
    int hipsolverDnDgesvd_bufferSize(Handle handle, int m, int n, int* lwork)
    int hipsolverDnCgesvd_bufferSize(Handle handle, int m, int n, int* lwork)
    int hipsolverDnZgesvd_bufferSize(Handle handle, int m, int n, int* lwork)

    int hipsolverDnSgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         float* A, int lda, float* S,
                         float* U, int ldu,
                         float* VT, int ldvt,
                         float* Work, int lwork,
                         float* rwork, int* devInfo)
    int hipsolverDnDgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         double* A, int lda, double* S,
                         double* U, int ldu,
                         double* VT, int ldvt,
                         double* Work, int lwork,
                         double* rwork, int* devInfo)
    int hipsolverDnCgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         rocblas_float_complex* A, int lda, float* S,
                         rocblas_float_complex* U, int ldu,
                         rocblas_float_complex* VT, int ldvt,
                         rocblas_float_complex* Work, int lwork,
                         float* rwork, int* devInfo)
    int hipsolverDnZgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         rocblas_double_complex* A, int lda, double* S,
                         rocblas_double_complex* U, int ldu,
                         rocblas_double_complex* VT, int ldvt,
                         rocblas_double_complex* Work, int lwork,
                         double* rwork, int* devInfo)

    # gesvdj ... Singular value decomposition using Jacobi mathod
    int hipsolverDnCreateGesvdjInfo(GesvdjInfo *info)
    int hipsolverDnDestroyGesvdjInfo(GesvdjInfo info)

    int hipsolverDnXgesvdjSetTolerance(GesvdjInfo info, double tolerance)
    int hipsolverDnXgesvdjSetMaxSweeps(GesvdjInfo info, int max_sweeps)
    int hipsolverDnXgesvdjSetSortEig(GesvdjInfo info, int sort_svd)
    int cusolverDnXgesvdjGetResidual(Handle handle, GesvdjInfo info,
                                     double* residual)
    int cusolverDnXgesvdjGetSweeps(Handle handle, GesvdjInfo info,
                                   int* executed_sweeps)

    int hipsolverDnSgesvdj_bufferSize(Handle handle, EigMode jobz, int econ,
                                     int m, int n, const float* A, int lda,
                                     const float* S, const float* U, int ldu,
                                     const float* V, int ldv, int* lwork,
                                     GesvdjInfo params)
    int hipsolverDnDgesvdj_bufferSize(Handle handle, EigMode jobz, int econ,
                                     int m, int n, const double* A, int lda,
                                     const double* S, const double* U, int ldu,
                                     const double* V, int ldv, int* lwork,
                                     GesvdjInfo params)
    int hipsolverDnCgesvdj_bufferSize(Handle handle, EigMode jobz, int econ,
                                     int m, int n, const rocblas_float_complex* A, int lda,
                                     const float* S, const rocblas_float_complex* U,
                                     int ldu, const rocblas_float_complex* V, int ldv,
                                     int* lwork, GesvdjInfo params)
    int hipsolverDnZgesvdj_bufferSize(Handle handle, EigMode jobz, int econ,
                                     int m, int n, const rocblas_double_complex* A,
                                     int lda, const double* S,
                                     const rocblas_double_complex* U, int ldu,
                                     const rocblas_double_complex* V, int ldv,
                                     int* lwork, GesvdjInfo params)

    int hipsolverDnSgesvdj(Handle handle, EigMode jobz, int econ, int m, int n,
                          float *A, int lda, float *S, float *U, int ldu,
                          float *V, int ldv, float *work, int lwork, int *info,
                          GesvdjInfo params)
    int hipsolverDnDgesvdj(Handle handle, EigMode jobz, int econ, int m, int n,
                          double *A, int lda, double *S, double *U, int ldu,
                          double *V, int ldv, double *work, int lwork,
                          int *info, GesvdjInfo params)
    int hipsolverDnCgesvdj(Handle handle, EigMode jobz, int econ, int m, int n,
                          rocblas_float_complex *A, int lda, float *S, rocblas_float_complex *U,
                          int ldu, rocblas_float_complex *V, int ldv, rocblas_float_complex *work,
                          int lwork, int *info, GesvdjInfo params)
    int hipsolverDnZgesvdj(Handle handle, EigMode jobz, int econ, int m, int n,
                          rocblas_double_complex *A, int lda, double *S,
                          rocblas_double_complex *U, int ldu, rocblas_double_complex *V,
                          int ldv, rocblas_double_complex *work, int lwork, int *info,
                          GesvdjInfo params)

    int hipsolverDnSgesvdjBatched_bufferSize(
        Handle handle, EigMode jobz, int m, int n, float* A, int lda,
        float* S, float* U, int ldu, float* V, int ldv,
        int* lwork, GesvdjInfo params, int batchSize)
    int hipsolverDnDgesvdjBatched_bufferSize(
        Handle handle, EigMode jobz, int m, int n, double* A, int lda,
        double* S, double* U, int ldu, double* V, int ldv,
        int* lwork, GesvdjInfo params, int batchSize)
    int hipsolverDnCgesvdjBatched_bufferSize(
        Handle handle, EigMode jobz, int m, int n, rocblas_float_complex* A, int lda,
        float* S, rocblas_float_complex* U, int ldu, rocblas_float_complex* V, int ldv,
        int* lwork, GesvdjInfo params, int batchSize)
    int hipsolverDnZgesvdjBatched_bufferSize(
        Handle handle, EigMode jobz, int m, int n, rocblas_double_complex* A, int lda,
        double* S, rocblas_double_complex* U, int ldu, rocblas_double_complex* V, int ldv,
        int* lwork, GesvdjInfo params, int batchSize)
    int hipsolverDnSgesvdjBatched(
        Handle handle, EigMode jobz, int m, int n, float* A, int lda, float* S,
        float* U, int ldu, float* V, int ldv, float* work, int lwork,
        int* info, GesvdjInfo params, int batchSize)
    int hipsolverDnDgesvdjBatched(
        Handle handle, EigMode jobz, int m, int n, double* A, int lda,
        double* S, double* U, int ldu, double* V, int ldv,
        double* work, int lwork,
        int* info, GesvdjInfo params, int batchSize)
    int hipsolverDnCgesvdjBatched(
        Handle handle, EigMode jobz, int m, int n, rocblas_float_complex* A, int lda,
        float* S, rocblas_float_complex* U, int ldu, rocblas_float_complex* V, int ldv,
        rocblas_float_complex* work, int lwork,
        int* info, GesvdjInfo params, int batchSize)
    int hipsolverDnZgesvdjBatched(
        Handle handle, EigMode jobz, int m, int n, rocblas_double_complex* A, int lda,
        double* S, rocblas_double_complex* U, int ldu, rocblas_double_complex* V, int ldv,
        rocblas_double_complex* work, int lwork,
        int* info, GesvdjInfo params, int batchSize)

    # gesvda ... Approximate singular value decomposition
    int hipsolverDnSgesvdaStridedBatched_bufferSize(
        Handle handle, EigMode jobz, int rank, int m, int n, const float *d_A,
        int lda, long long int strideA, const float *d_S,
        long long int strideS, const float *d_U, int ldu,
        long long int strideU, const float *d_V, int ldv,
        long long int strideV, int *lwork, int batchSize)

    int hipsolverDnDgesvdaStridedBatched_bufferSize(
        Handle handle, EigMode jobz, int rank, int m, int n, const double *d_A,
        int lda, long long int strideA, const double *d_S,
        long long int strideS, const double *d_U, int ldu,
        long long int strideU, const double *d_V, int ldv,
        long long int strideV, int *lwork, int batchSize)

    int hipsolverDnCgesvdaStridedBatched_bufferSize(
        Handle handle, EigMode jobz, int rank, int m, int n,
        const rocblas_float_complex *d_A, int lda, long long int strideA, const float *d_S,
        long long int strideS, const rocblas_float_complex *d_U, int ldu,
        long long int strideU, const rocblas_float_complex *d_V, int ldv,
        long long int strideV, int *lwork, int batchSize)

    int hipsolverDnZgesvdaStridedBatched_bufferSize(
        Handle handle, EigMode jobz, int rank, int m, int n,
        const rocblas_double_complex *d_A, int lda, long long int strideA,
        const double *d_S, long long int strideS, const rocblas_double_complex *d_U,
        int ldu, long long int strideU, const rocblas_double_complex *d_V, int ldv,
        long long int strideV, int *lwork, int batchSize)

    int hipsolverDnSgesvdaStridedBatched(
        Handle handle, EigMode jobz, int rank, int m, int n, const float *d_A,
        int lda, long long int strideA, float *d_S, long long int strideS,
        float *d_U, int ldu, long long int strideU, float *d_V, int ldv,
        long long int strideV, float *d_work, int lwork, int *d_info,
        double *h_R_nrmF, int batchSize)

    int hipsolverDnDgesvdaStridedBatched(
        Handle handle, EigMode jobz, int rank, int m, int n, const double *d_A,
        int lda, long long int strideA, double *d_S, long long int strideS,
        double *d_U, int ldu, long long int strideU, double *d_V, int ldv,
        long long int strideV, double *d_work, int lwork, int *d_info,
        double *h_R_nrmF, int batchSize)

    int hipsolverDnCgesvdaStridedBatched(
        Handle handle, EigMode jobz, int rank, int m, int n,
        const rocblas_float_complex *d_A, int lda, long long int strideA, float *d_S,
        long long int strideS, rocblas_float_complex *d_U, int ldu, long long int strideU,
        rocblas_float_complex *d_V, int ldv, long long int strideV, rocblas_float_complex *d_work,
        int lwork, int *d_info, double *h_R_nrmF, int batchSize)

    int hipsolverDnZgesvdaStridedBatched(
        Handle handle, EigMode jobz, int rank, int m, int n,
        const rocblas_double_complex *d_A, int lda, long long int strideA,
        double *d_S, long long int strideS, rocblas_double_complex *d_U, int ldu,
        long long int strideU, rocblas_double_complex *d_V, int ldv,
        long long int strideV, rocblas_double_complex *d_work, int lwork, int *d_info,
        double *h_R_nrmF, int batchSize)

    # Standard symmetric eigenvalue solver
    int hipsolverDnSsyevd_bufferSize(Handle handle,
                                    EigMode jobz, FillMode uplo, int n,
                                    const float* A, int lda,
                                    const float* W, int* lwork)
    int hipsolverDnDsyevd_bufferSize(Handle handle,
                                    EigMode jobz, FillMode uplo, int n,
                                    const double* A, int lda,
                                    const double* W, int* lwork)
    int hipsolverDnCheevd_bufferSize(Handle handle,
                                    EigMode jobz, FillMode uplo, int n,
                                    const rocblas_float_complex* A, int lda,
                                    const float* W, int* lwork)
    int hipsolverDnZheevd_bufferSize(Handle handle,
                                    EigMode jobz, FillMode uplo, int n,
                                    const rocblas_double_complex* A, int lda,
                                    const double* W, int* lwork)

    int hipsolverDnSsyevd(Handle handle, EigMode jobz, FillMode uplo, int n,
                         float* A, int lda, float* W,
                         float* work, int lwork, int* info)
    int hipsolverDnDsyevd(Handle handle, EigMode jobz, FillMode uplo, int n,
                         double* A, int lda, double* W,
                         double* work, int lwork, int* info)
    int hipsolverDnCheevd(Handle handle, EigMode jobz, FillMode uplo, int n,
                         rocblas_float_complex* A, int lda, float* W,
                         rocblas_float_complex* work, int lwork, int* info)
    int hipsolverDnZheevd(Handle handle, EigMode jobz, FillMode uplo, int n,
                         rocblas_double_complex* A, int lda, double* W,
                         rocblas_double_complex* work, int lwork, int* info)

    # Symmetric eigenvalue solver using Jacobi method
    int hipsolverDnCreateSyevjInfo(SyevjInfo *info)
    int hipsolverDnDestroySyevjInfo(SyevjInfo info)

    int cusolverDnXsyevjSetTolerance(SyevjInfo info, double tolerance)
    int cusolverDnXsyevjSetMaxSweeps(SyevjInfo info, int max_sweeps)
    int hipsolverDnXsyevjSetSortEig(SyevjInfo info, int sort_eig)
    int cusolverDnXsyevjGetResidual(
        Handle handle, SyevjInfo info, double* residual)
    int cusolverDnXsyevjGetSweeps(
        Handle handle, SyevjInfo info, int* executed_sweeps)

    int hipsolverDnSsyevj_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const float *A, int lda, const float *W, int *lwork,
        SyevjInfo params)
    int hipsolverDnDsyevj_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const double *A, int lda, const double *W, int *lwork,
        SyevjInfo params)
    int hipsolverDnCheevj_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const rocblas_float_complex *A, int lda, const float *W, int *lwork,
        SyevjInfo params)
    int hipsolverDnZheevj_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const rocblas_double_complex *A, int lda, const double *W, int *lwork,
        SyevjInfo params)

    int hipsolverDnSsyevj(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        float *A, int lda, float *W, float *work,
        int lwork, int *info, SyevjInfo params)
    int hipsolverDnDsyevj(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        double *A, int lda, double *W, double *work,
        int lwork, int *info, SyevjInfo params)
    int hipsolverDnCheevj(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        rocblas_float_complex *A, int lda, float *W, rocblas_float_complex *work,
        int lwork, int *info, SyevjInfo params)
    int hipsolverDnZheevj(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        rocblas_double_complex *A, int lda, double *W, rocblas_double_complex *work,
        int lwork, int *info, SyevjInfo params)

    int hipsolverDnSsyevjBatched_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const float *A, int lda, const float *W, int *lwork,
        SyevjInfo params, int batchSize)

    int hipsolverDnDsyevjBatched_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const double *A, int lda, const double *W, int *lwork,
        SyevjInfo params, int batchSize)

    int hipsolverDnCheevjBatched_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const rocblas_float_complex *A, int lda, const float *W, int *lwork,
        SyevjInfo params, int batchSize)

    int hipsolverDnZheevjBatched_bufferSize(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        const rocblas_double_complex *A, int lda, const double *W, int *lwork,
        SyevjInfo params, int batchSize)

    int hipsolverDnSsyevjBatched(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        float *A, int lda, float *W, float *work, int lwork,
        int *info, SyevjInfo params, int batchSize)

    int hipsolverDnDsyevjBatched(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        double *A, int lda, double *W, double *work, int lwork,
        int *info, SyevjInfo params, int batchSize)

    int hipsolverDnCheevjBatched(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        rocblas_float_complex *A, int lda, float *W, rocblas_float_complex *work, int lwork,
        int *info, SyevjInfo params, int batchSize)

    int hipsolverDnZheevjBatched(
        Handle handle, EigMode jobz, FillMode uplo, int n,
        rocblas_double_complex *A, int lda, double *W, rocblas_double_complex *work,
        int lwork, int *info, SyevjInfo params, int batchSize)

    # 64bit
    int hipsolverDnXsyevd_bufferSize(
        Handle handle, Params params, EigMode jobz, FillMode uplo, int64_t n,
        DataType dataTypeA, void *A, int64_t lda,
        DataType dataTypeW, void *W, DataType computeType,
        size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost)
    int hipsolverDnXsyevd(
        Handle handle, Params params, EigMode jobz, FillMode uplo, int64_t n,
        DataType dataTypeA, void *A, int64_t lda,
        DataType dataTypeW, void *W, DataType computeType,
        void *bufferOnDevice, size_t workspaceInBytesOnDevice,
        void *bufferOnHost, size_t workspaceInBytesOnHost, int *info)

    ###########################################################################
    # Sparse LAPACK Functions
    ###########################################################################

    int cusolverSpScsrlsvchol(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const float* b, float tol, int reorder, float* x, int* singularity)
    int cusolverSpDcsrlsvchol(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const double* b, double tol, int reorder, double* x, int* singularity)
    int cusolverSpCcsrlsvchol(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const rocblas_float_complex *csrVal,
        const int *csrRowPtr, const int *csrColInd, const rocblas_float_complex *b,
        float tol, int reorder, rocblas_float_complex *x, int *singularity)
    int cusolverSpZcsrlsvchol(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const rocblas_double_complex *csrVal,
        const int *csrRowPtr, const int *csrColInd, const rocblas_double_complex *b,
        double tol, int reorder, rocblas_double_complex *x, int *singularity)

    int cusolverSpScsrlsvqr(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const float* b, float tol, int reorder, float* x, int* singularity)
    int cusolverSpDcsrlsvqr(
        SpHandle handle, int m, int nnz, const MatDescr descrA,
        const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
        const double* b, double tol, int reorder, double* x, int* singularity)
    int cusolverSpCcsrlsvqr(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const rocblas_float_complex *csrVal,
        const int *csrRowPtr, const int *csrColInd, const rocblas_float_complex *b,
        float tol, int reorder, rocblas_float_complex *x, int *singularity)
    int cusolverSpZcsrlsvqr(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const rocblas_double_complex *csrVal,
        const int *csrRowPtr, const int *csrColInd, const rocblas_double_complex *b,
        double tol, int reorder, rocblas_double_complex *x, int *singularity)

    int cusolverSpScsreigvsi(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const float *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, float mu0,
        const float *x0, int maxite, float eps, float *mu, float *x)
    int cusolverSpDcsreigvsi(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const double *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, double mu0,
        const double *x0, int maxite, double eps, double *mu, double *x)
    int cusolverSpCcsreigvsi(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const rocblas_float_complex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, rocblas_float_complex mu0,
        const rocblas_float_complex *x0, int maxite, float eps, rocblas_float_complex *mu,
        rocblas_float_complex *x)
    int cusolverSpZcsreigvsi(
        SpHandle handle, int m, int nnz,
        const MatDescr descrA, const rocblas_double_complex *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, rocblas_double_complex mu0,
        const rocblas_double_complex *x0, int maxite, double eps, rocblas_double_complex *mu,
        rocblas_double_complex *x)

###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    0: 'CUSOLVER_STATUS_SUCCESS',
    1: 'CUSOLVER_STATUS_NOT_INITIALIZED',
    2: 'CUSOLVER_STATUS_ALLOC_FAILED',
    3: 'CUSOLVER_STATUS_INVALID_VALUE',
    4: 'CUSOLVER_STATUS_ARCH_MISMATCH',
    5: 'CUSOLVER_STATUS_MAPPING_ERROR',
    6: 'CUSOLVER_STATUS_EXECUTION_FAILED',
    7: 'CUSOLVER_STATUS_INTERNAL_ERROR',
    8: 'CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED',
    9: 'CUSOLVER_STATUS_NOT_SUPPORTED',
    10: 'CUSOLVER_STATUS_ZERO_PIVOT',
    11: 'CUSOLVER_STATUS_INVALID_LICENSE',
    12: 'CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED',
    13: 'CUSOLVER_STATUS_IRS_PARAMS_INVALID',
    14: 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC',
    15: 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE',
    16: 'CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER',
    20: 'CUSOLVER_STATUS_IRS_INTERNAL_ERROR',
    21: 'CUSOLVER_STATUS_IRS_NOT_SUPPORTED',
    22: 'CUSOLVER_STATUS_IRS_OUT_OF_RANGE',
    23: 'CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES',
    25: 'CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED',
    26: 'CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED',
    30: 'CUSOLVER_STATUS_IRS_MATRIX_SINGULAR',
    31: 'CUSOLVER_STATUS_INVALID_WORKSPACE',
}

# for rocBLAS and rocSOLVER
cdef dict ROC_STATUS = {
    0: 'rocblas_status_success',
    1: 'rocblas_status_invalid_handle',
    2: 'rocblas_status_not_implemented',
    3: 'rocblas_status_invalid_pointer',
    4: 'rocblas_status_invalid_size',
    5: 'rocblas_status_memory_error',
    6: 'rocblas_status_internal_error',
    7: 'rocblas_status_perf_degraded',
    8: 'rocblas_status_size_query_mismatch',
    9: 'rocblas_status_size_increased',
    10: 'rocblas_status_size_unchanged',
    11: 'rocblas_status_invalid_value',
    12: 'rocblas_status_continue',
}


class CUSOLVERError(RuntimeError):

    def __init__(self, status):
        self.status = status
        if runtime._is_hip_environment:
            err = ROC_STATUS
        else:
            err = STATUS
        super(CUSOLVERError, self).__init__(err[status])

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUSOLVERError(status)


###############################################################################
# Library Attributes
###############################################################################

cpdef int getProperty(int type) except? -1:
    cdef int value
    with nogil:
        status = cusolverGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef tuple _getVersion():
    return (getProperty(hipLibraryMajorVersion),
            getProperty(hipLibraryMinorVersion),
            getProperty(hipLibraryPatchVersion))


###############################################################################
# Context
###############################################################################

cpdef intptr_t create() except? 0:
    cdef Handle handle
    with nogil:
        status = hipsolverDnCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef intptr_t spCreate() except? 0:
    cdef SpHandle handle
    with nogil:
        status = cusolverSpCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    with nogil:
        status = hipsolverDnDestroy(<Handle>handle)
    check_status(status)


cpdef spDestroy(intptr_t handle):
    with nogil:
        status = cusolverSpDestroy(<SpHandle>handle)
    check_status(status)


###############################################################################
# Stream
###############################################################################

cpdef setStream(intptr_t handle, size_t stream):
    # TODO(leofang): The support of stream capture is not mentioned at all in
    # the cuSOLVER docs (as of CUDA 11.5), so we disable this functionality.
    if not runtime._is_hip_environment and runtime.streamIsCapturing(stream):
        raise NotImplementedError(
            'calling cuSOLVER API during stream capture is currently '
            'unsupported')

    with nogil:
        status = hipsolverDnSetStream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef size_t getStream(intptr_t handle) except? 0:
    cdef Stream stream
    with nogil:
        status = cusolverDnGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


cpdef spSetStream(intptr_t handle, size_t stream):
    with nogil:
        status = cusolverSpSetStream(<SpHandle>handle, <Stream>stream)
    check_status(status)


cpdef size_t spGetStream(intptr_t handle) except *:
    cdef Stream stream
    with nogil:
        status = cusolverSpGetStream(<SpHandle>handle, &stream)
    check_status(status)
    return <size_t>stream


cdef _setStream(intptr_t handle):
    """Set current stream"""
    setStream(handle, stream_module.get_current_stream_ptr())


cdef _spSetStream(intptr_t handle):
    """Set current stream"""
    spSetStream(handle, stream_module.get_current_stream_ptr())


###############################################################################
# Params
###############################################################################

cpdef intptr_t createParams() except? 0:
    cdef Params params
    with nogil:
        status = cusolverDnCreateParams(&params)
    check_status(status)
    return <intptr_t>params

cpdef destroyParams(intptr_t params):
    with nogil:
        status = cusolverDnDestroyParams(<Params>params)
    check_status(status)


###########################################################################
# Dense LAPACK Functions (Linear Solver)
###########################################################################

# Cholesky factorization
cpdef int spotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnSpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <float*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int dpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnDpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <double*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int cpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnCpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <rocblas_float_complex*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef int zpotrf_bufferSize(intptr_t handle, int uplo,
                            int n, size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnZpotrf_bufferSize(
            <Handle>handle, <FillMode>uplo, n,
            <rocblas_double_complex*>A, <int>lda, &lwork)
    check_status(status)
    return lwork

cpdef spotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSpotrf(
            <Handle>handle, <FillMode>uplo, n, <float*>A,
            lda, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDpotrf(
            <Handle>handle, <FillMode>uplo, n, <double*>A,
            lda, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef cpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCpotrf(
            <Handle>handle, <FillMode>uplo, n, <rocblas_float_complex*>A,
            lda, <rocblas_float_complex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zpotrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZpotrf(
            <Handle>handle, <FillMode>uplo, n, <rocblas_double_complex*>A,
            lda, <rocblas_double_complex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef spotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const float*>A, lda, <float*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef dpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const double*>A, lda, <double*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef cpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const rocblas_float_complex*>A, lda, <rocblas_float_complex*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef zpotrs(intptr_t handle, int uplo, int n, int nrhs,
             size_t A, int lda, size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZpotrs(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <const rocblas_double_complex*>A, lda, <rocblas_double_complex*>B, ldb,
            <int*>devInfo)
    check_status(status)

cpdef spotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnSpotrfBatched(
            <Handle>handle, <FillMode>uplo, n, <float**>Aarray,
            lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef dpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnDpotrfBatched(
            <Handle>handle, <FillMode>uplo, n, <double**>Aarray,
            lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef cpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnCpotrfBatched(
            <Handle>handle, <FillMode>uplo, n, <rocblas_float_complex**>Aarray,
            lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef zpotrfBatched(intptr_t handle, int uplo, int n, size_t Aarray, int lda,
                    size_t infoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnZpotrfBatched(
            <Handle>handle, <FillMode>uplo, n, <rocblas_double_complex**>Aarray,
            lda, <int*>infoArray, batchSize)
    check_status(status)

cpdef spotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnSpotrsBatched(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <float**>Aarray, lda, <float**>Barray, ldb,
            <int*>devInfo, batchSize)
    check_status(status)

cpdef dpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnDpotrsBatched(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <double**>Aarray, lda, <double**>Barray, ldb,
            <int*>devInfo, batchSize)
    check_status(status)

cpdef cpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnCpotrsBatched(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <rocblas_float_complex**>Aarray, lda, <rocblas_float_complex**>Barray, ldb,
            <int*>devInfo, batchSize)
    check_status(status)

cpdef zpotrsBatched(intptr_t handle, int uplo, int n, int nrhs, size_t Aarray,
                    int lda, size_t Barray, int ldb, size_t devInfo,
                    int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnZpotrsBatched(
            <Handle>handle, <FillMode>uplo, n, nrhs,
            <rocblas_double_complex**>Aarray, lda, <rocblas_double_complex**>Barray, ldb,
            <int*>devInfo, batchSize)
    check_status(status)

# LU factorization
cpdef int sgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgetrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgetrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int cgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgetrf_bufferSize(
            <Handle>handle, m, n, <rocblas_float_complex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zgetrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgetrf_bufferSize(
            <Handle>handle, m, n, <rocblas_double_complex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgetrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef dgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgetrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef cgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgetrf(
            <Handle>handle, m, n, <rocblas_float_complex*>A, lda,
            <rocblas_float_complex*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)

cpdef zgetrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t work, size_t devIpiv, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgetrf(
            <Handle>handle, m, n, <rocblas_double_complex*>A, lda,
            <rocblas_double_complex*>work, <int*>devIpiv, <int*>devInfo)
    check_status(status)


# LU solve
cpdef sgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const float*> A, lda, <const int*>devIpiv,
            <float*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef dgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const double*> A, lda, <const int*>devIpiv,
            <double*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef cgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const rocblas_float_complex*> A, lda, <const int*>devIpiv,
            <rocblas_float_complex*>B, ldb, <int*> devInfo)
    check_status(status)

cpdef zgetrs(intptr_t handle, int trans, int n, int nrhs,
             size_t A, int lda, size_t devIpiv,
             size_t B, int ldb, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgetrs(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const rocblas_double_complex*> A, lda, <const int*>devIpiv,
            <rocblas_double_complex*>B, ldb, <int*> devInfo)
    check_status(status)


# QR factorization
cpdef int sgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgeqrf_bufferSize(
            <Handle>handle, m, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgeqrf_bufferSize(
            <Handle>handle, m, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int cgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgeqrf_bufferSize(
            <Handle>handle, m, n, <rocblas_float_complex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zgeqrf_bufferSize(intptr_t handle, int m, int n,
                            size_t A, int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgeqrf_bufferSize(
            <Handle>handle, m, n, <rocblas_double_complex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef sgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgeqrf(
            <Handle>handle, m, n, <float*>A, lda,
            <float*>tau, <float*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef dgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgeqrf(
            <Handle>handle, m, n, <double*>A, lda,
            <double*>tau, <double*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef cgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgeqrf(
            <Handle>handle, m, n, <rocblas_float_complex*>A, lda,
            <rocblas_float_complex*>tau, <rocblas_float_complex*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef zgeqrf(intptr_t handle, int m, int n, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgeqrf(
            <Handle>handle, m, n, <rocblas_double_complex*>A, lda,
            <rocblas_double_complex*>tau, <rocblas_double_complex*>work, lwork,
            <int*>devInfo)
    check_status(status)


# Generate unitary matrix Q from QR factorization
cpdef int sorgqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnSorgqr_bufferSize(
            <Handle>handle, m, n, k, <const float*>A, lda,
            <const float*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int dorgqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnDorgqr_bufferSize(
            <Handle>handle, m, n, k, <const double*>A, lda,
            <const double*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int cungqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnCungqr_bufferSize(
            <Handle>handle, m, n, k, <const rocblas_float_complex*>A, lda,
            <const rocblas_float_complex*>tau, &lwork)
    check_status(status)
    return lwork

cpdef int zungqr_bufferSize(intptr_t handle, int m, int n, int k,
                            size_t A, int lda, size_t tau) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnZungqr_bufferSize(
            <Handle>handle, m, n, k, <const rocblas_double_complex*>A, lda,
            <const rocblas_double_complex*>tau, &lwork)
    check_status(status)
    return lwork

cpdef sorgqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSorgqr(
            <Handle>handle, m, n, k, <float*>A, lda,
            <const float*>tau, <float*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef dorgqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDorgqr(
            <Handle>handle, m, n, k, <double*>A, lda,
            <const double*>tau, <double*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef cungqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCungqr(
            <Handle>handle, m, n, k, <rocblas_float_complex*>A, lda,
            <const rocblas_float_complex*>tau, <rocblas_float_complex*>work, lwork,
            <int*>devInfo)
    check_status(status)

cpdef zungqr(intptr_t handle, int m, int n, int k, size_t A, int lda,
             size_t tau, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZungqr(
            <Handle>handle, m, n, k, <rocblas_double_complex*>A, lda,
            <const rocblas_double_complex*>tau, <rocblas_double_complex*>work, lwork,
            <int*>devInfo)
    check_status(status)


# Compute Q**T*b in solve min||A*x = b||
cpdef int sormqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnSormqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const float*>A, lda, <const float*>tau,
            <float*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int dormqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnDormqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const double*>A, lda, <const double*>tau,
            <double*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int cunmqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnCunmqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const rocblas_float_complex*>A, lda, <const rocblas_float_complex*>tau,
            <rocblas_float_complex*>C, ldc, &lwork)
    check_status(status)
    return lwork

cpdef int zunmqr_bufferSize(intptr_t handle, int side, int trans,
                            int m, int n, int k, size_t A, int lda, size_t tau,
                            size_t C, int ldc) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnZunmqr_bufferSize(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const rocblas_double_complex*>A, lda, <const rocblas_double_complex*>tau,
            <rocblas_double_complex*>C, ldc, &lwork)
    check_status(status)
    return lwork


cpdef sormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const float*>A, lda, <const float*>tau,
            <float*>C, ldc,
            <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDormqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const double*>A, lda, <const double*>tau,
            <double*>C, ldc,
            <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef cunmqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCunmqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const rocblas_float_complex*>A, lda, <const rocblas_float_complex*>tau,
            <rocblas_float_complex*>C, ldc,
            <rocblas_float_complex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zunmqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZunmqr(
            <Handle>handle, <SideMode>side, <Operation>trans, m, n, k,
            <const rocblas_double_complex*>A, lda, <const rocblas_double_complex*>tau,
            <rocblas_double_complex*>C, ldc,
            <rocblas_double_complex*>work, lwork, <int*>devInfo)
    check_status(status)

# (obsoleted)
cpdef cormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    return cunmqr(handle, side, trans, m, n, k, A, lda, tau,
                  C, ldc, work, lwork, devInfo)

# (obsoleted)
cpdef zormqr(intptr_t handle, int side, int trans,
             int m, int n, int k, size_t A, int lda, size_t tau,
             size_t C, int ldc, size_t work, int lwork, size_t devInfo):
    return zunmqr(handle, side, trans, m, n, k, A, lda, tau,
                  C, ldc, work, lwork, devInfo)


# L*D*L**T,U*D*U**T factorization
cpdef int ssytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnSsytrf_bufferSize(
            <Handle>handle, n, <float*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int dsytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnDsytrf_bufferSize(
            <Handle>handle, n, <double*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int csytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnCsytrf_bufferSize(
            <Handle>handle, n, <rocblas_float_complex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef int zsytrf_bufferSize(intptr_t handle, int n, size_t A,
                            int lda) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnZsytrf_bufferSize(
            <Handle>handle, n, <rocblas_double_complex*>A, lda, &lwork)
    check_status(status)
    return lwork

cpdef ssytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSsytrf(
            <Handle>handle, <FillMode>uplo, n, <float*>A, lda,
            <int*>ipiv, <float*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef dsytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDsytrf(
            <Handle>handle, <FillMode>uplo, n, <double*>A, lda,
            <int*>ipiv, <double*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef csytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCsytrf(
            <Handle>handle, <FillMode>uplo, n, <rocblas_float_complex*>A, lda,
            <int*>ipiv, <rocblas_float_complex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef zsytrf(intptr_t handle, int uplo, int n, size_t A, int lda,
             size_t ipiv, size_t work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZsytrf(
            <Handle>handle, <FillMode>uplo, n, <rocblas_double_complex*>A, lda,
            <int*>ipiv, <rocblas_double_complex*>work, lwork, <int*>devInfo)
    check_status(status)

cpdef size_t zzgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZZgesv_bufferSize(
            <Handle>handle, n, nrhs, <rocblas_double_complex*>dA, ldda, <int*>dipiv,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zcgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZCgesv_bufferSize(
            <Handle>handle, n, nrhs, <rocblas_double_complex*>dA, ldda, <int*>dipiv,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zygesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZYgesv_bufferSize(
            <Handle>handle, n, nrhs, <rocblas_double_complex*>dA, ldda, <int*>dipiv,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zkgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZKgesv_bufferSize(
            <Handle>handle, n, nrhs, <rocblas_double_complex*>dA, ldda, <int*>dipiv,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ccgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCCgesv_bufferSize(
            <Handle>handle, n, nrhs, <rocblas_float_complex*>dA, ldda, <int*>dipiv,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t cygesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCYgesv_bufferSize(
            <Handle>handle, n, nrhs, <rocblas_float_complex*>dA, ldda, <int*>dipiv,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ckgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCKgesv_bufferSize(
            <Handle>handle, n, nrhs, <rocblas_float_complex*>dA, ldda, <int*>dipiv,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ddgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDDgesv_bufferSize(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dsgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDSgesv_bufferSize(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dxgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDXgesv_bufferSize(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dhgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDHgesv_bufferSize(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ssgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSSgesv_bufferSize(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t sxgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSXgesv_bufferSize(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t shgesv_bufferSize(intptr_t handle, int n, int nrhs, size_t dA,
                               int ldda, size_t dipiv, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSHgesv_bufferSize(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef int zzgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZZgesv(
            <Handle>handle, n, nrhs, <rocblas_double_complex*>dA, ldda, <int*>dipiv,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zcgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZCgesv(
            <Handle>handle, n, nrhs, <rocblas_double_complex*>dA, ldda, <int*>dipiv,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zygesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZYgesv(
            <Handle>handle, n, nrhs, <rocblas_double_complex*>dA, ldda, <int*>dipiv,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zkgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZKgesv(
            <Handle>handle, n, nrhs, <rocblas_double_complex*>dA, ldda, <int*>dipiv,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ccgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCCgesv(
            <Handle>handle, n, nrhs, <rocblas_float_complex*>dA, ldda, <int*>dipiv,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int cygesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCYgesv(
            <Handle>handle, n, nrhs, <rocblas_float_complex*>dA, ldda, <int*>dipiv,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ckgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCKgesv(
            <Handle>handle, n, nrhs, <rocblas_float_complex*>dA, ldda, <int*>dipiv,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ddgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDDgesv(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dsgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDSgesv(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dxgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDXgesv(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dhgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDHgesv(
            <Handle>handle, n, nrhs, <double*>dA, ldda, <int*>dipiv,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ssgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSSgesv(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int sxgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSXgesv(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int shgesv(intptr_t handle, int n, int nrhs, size_t dA, int ldda,
                 size_t dipiv, size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSHgesv(
            <Handle>handle, n, nrhs, <float*>dA, ldda, <int*>dipiv,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef size_t zzgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZZgels_bufferSize(
            <Handle>handle, m, n, nrhs, <rocblas_double_complex*>dA, ldda,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zcgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZCgels_bufferSize(
            <Handle>handle, m, n, nrhs, <rocblas_double_complex*>dA, ldda,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zygels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZYgels_bufferSize(
            <Handle>handle, m, n, nrhs, <rocblas_double_complex*>dA, ldda,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t zkgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZKgels_bufferSize(
            <Handle>handle, m, n, nrhs, <rocblas_double_complex*>dA, ldda,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ccgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCCgels_bufferSize(
            <Handle>handle, m, n, nrhs, <rocblas_float_complex*>dA, ldda,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t cygels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCYgels_bufferSize(
            <Handle>handle, m, n, nrhs, <rocblas_float_complex*>dA, ldda,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ckgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCKgels_bufferSize(
            <Handle>handle, m, n, nrhs, <rocblas_float_complex*>dA, ldda,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ddgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDDgels_bufferSize(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dsgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDSgels_bufferSize(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dxgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDXgels_bufferSize(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t dhgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDHgels_bufferSize(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t ssgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSSgels_bufferSize(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t sxgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSXgels_bufferSize(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef size_t shgels_bufferSize(intptr_t handle, int m, int n, int nrhs,
                               size_t dA, int ldda, size_t dB, int lddb,
                               size_t dX, int lddx, size_t dwork) except? -1:
    cdef size_t lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSHgels_bufferSize(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx, <void*>dwork, &lwork)
    check_status(status)
    return lwork

cpdef int zzgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZZgels(
            <Handle>handle, m, n, nrhs, <rocblas_double_complex*>dA, ldda,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zcgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZCgels(
            <Handle>handle, m, n, nrhs, <rocblas_double_complex*>dA, ldda,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zygels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZYgels(
            <Handle>handle, m, n, nrhs, <rocblas_double_complex*>dA, ldda,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int zkgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnZKgels(
            <Handle>handle, m, n, nrhs, <rocblas_double_complex*>dA, ldda,
            <rocblas_double_complex*>dB, lddb, <rocblas_double_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ccgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCCgels(
            <Handle>handle, m, n, nrhs, <rocblas_float_complex*>dA, ldda,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int cygels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCYgels(
            <Handle>handle, m, n, nrhs, <rocblas_float_complex*>dA, ldda,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ckgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnCKgels(
            <Handle>handle, m, n, nrhs, <rocblas_float_complex*>dA, ldda,
            <rocblas_float_complex*>dB, lddb, <rocblas_float_complex*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ddgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDDgels(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dsgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDSgels(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dxgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDXgels(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int dhgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnDHgels(
            <Handle>handle, m, n, nrhs, <double*>dA, ldda,
            <double*>dB, lddb, <double*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int ssgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSSgels(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int sxgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSXgels(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

cpdef int shgels(intptr_t handle, int m, int n, int nrhs, size_t dA, int ldda,
                 size_t dB, int lddb, size_t dX, int lddx,
                 size_t dwork, size_t lwork, size_t dInfo):
    cdef int iter
    _setStream(handle)
    with nogil:
        status = cusolverDnSHgels(
            <Handle>handle, m, n, nrhs, <float*>dA, ldda,
            <float*>dB, lddb, <float*>dX, lddx,
            <void*>dwork, lwork, &iter, <int*>dInfo)
    check_status(status)
    return iter

###############################################################################
# Dense LAPACK Functions (Eigenvalue Solver)
###############################################################################

# Bidiagonal factorization
cpdef int sgebrd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnSgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgebrd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnDgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int cgebrd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnCgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int zgebrd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = cusolverDnZgebrd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnSgebrd(
            <Handle>handle, m, n,
            <float*>A, lda,
            <float*>D, <float*>E,
            <float*>tauQ, <float*>tauP,
            <float*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef dgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnDgebrd(
            <Handle>handle, m, n,
            <double*>A, lda,
            <double*>D, <double*>E,
            <double*>tauQ, <double*>tauP,
            <double*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef cgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnCgebrd(
            <Handle>handle, m, n,
            <rocblas_float_complex*>A, lda,
            <float*>D, <float*>E,
            <rocblas_float_complex*>tauQ, <rocblas_float_complex*>tauP,
            <rocblas_float_complex*>Work, lwork, <int*>devInfo)
    check_status(status)

cpdef zgebrd(intptr_t handle, int m, int n, size_t A, int lda,
             size_t D, size_t E, size_t tauQ, size_t tauP,
             size_t Work, int lwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = cusolverDnZgebrd(
            <Handle>handle, m, n,
            <rocblas_double_complex*>A, lda,
            <double*>D, <double*>E,
            <rocblas_double_complex*>tauQ, <rocblas_double_complex*>tauP,
            <rocblas_double_complex*>Work, lwork, <int*>devInfo)
    check_status(status)


# Singular value decomposition, A = U * Sigma * V^H
cpdef int sgesvd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int dgesvd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int cgesvd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef int zgesvd_bufferSize(intptr_t handle, int m, int n) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgesvd_bufferSize(<Handle>handle, m, n, &lwork)
    check_status(status)
    return lwork

cpdef sgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgesvd(
            <Handle>handle, jobu, jobvt, m, n, <float*>A, lda,
            <float*>S, <float*>U, ldu, <float*>VT, ldvt,
            <float*>Work, lwork, <float*>rwork, <int*>devInfo)
    check_status(status)

cpdef dgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgesvd(
            <Handle>handle, jobu, jobvt, m, n, <double*>A, lda,
            <double*>S, <double*>U, ldu, <double*>VT, ldvt,
            <double*>Work, lwork, <double*>rwork, <int*>devInfo)
    check_status(status)

cpdef cgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgesvd(
            <Handle>handle, jobu, jobvt, m, n, <rocblas_float_complex*>A, lda,
            <float*>S, <rocblas_float_complex*>U, ldu, <rocblas_float_complex*>VT, ldvt,
            <rocblas_float_complex*>Work, lwork, <float*>rwork, <int*>devInfo)
    check_status(status)

cpdef zgesvd(intptr_t handle, char jobu, char jobvt, int m, int n, size_t A,
             int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
             size_t Work, int lwork, size_t rwork, size_t devInfo):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgesvd(
            <Handle>handle, jobu, jobvt, m, n, <rocblas_double_complex*>A, lda,
            <double*>S, <rocblas_double_complex*>U, ldu, <rocblas_double_complex*>VT, ldvt,
            <rocblas_double_complex*>Work, lwork, <double*>rwork, <int*>devInfo)
    check_status(status)

# gesvdj ... Singular value decomposition using Jacobi mathod
cpdef intptr_t createGesvdjInfo() except? 0:
    cdef GesvdjInfo info
    status = hipsolverDnCreateGesvdjInfo(&info)
    check_status(status)
    return <intptr_t>info

cpdef destroyGesvdjInfo(intptr_t info):
    status = hipsolverDnDestroyGesvdjInfo(<GesvdjInfo>info)
    check_status(status)

cpdef xgesvdjSetTolerance(intptr_t info, double tolerance):
    status = hipsolverDnXgesvdjSetTolerance(<GesvdjInfo>info, tolerance)
    check_status(status)

cpdef xgesvdjSetMaxSweeps(intptr_t info, int max_sweeps):
    status = hipsolverDnXgesvdjSetMaxSweeps(<GesvdjInfo>info, max_sweeps)
    check_status(status)

cpdef xgesvdjSetSortEig(intptr_t info, int sort_svd):
    status = hipsolverDnXgesvdjSetSortEig(<GesvdjInfo>info, sort_svd)
    check_status(status)

cpdef double xgesvdjGetResidual(intptr_t handle, intptr_t info):
    cdef double residual
    status = cusolverDnXgesvdjGetResidual(<Handle>handle, <GesvdjInfo>info,
                                          &residual)
    check_status(status)
    return residual

cpdef int xgesvdjGetSweeps(intptr_t handle, intptr_t info):
    cdef int executed_sweeps
    status = cusolverDnXgesvdjGetSweeps(<Handle>handle, <GesvdjInfo>info,
                                        &executed_sweeps)
    check_status(status)
    return executed_sweeps

cpdef int sgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params):
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgesvdj_bufferSize(
            <Handle>handle, <EigMode>jobz, econ, m, n, <const float*>A, lda,
            <const float*>S, <const float*>U, ldu, <const float*>V, ldv,
            &lwork, <GesvdjInfo>params)
    check_status(status)
    return lwork

cpdef int dgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params):
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgesvdj_bufferSize(
            <Handle>handle, <EigMode>jobz, econ, m, n, <const double*>A, lda,
            <const double*>S, <const double*>U, ldu, <const double*>V, ldv,
            &lwork, <GesvdjInfo>params)
    check_status(status)
    return lwork

cpdef int cgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params):
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgesvdj_bufferSize(
            <Handle>handle, <EigMode>jobz, econ, m, n, <const rocblas_float_complex*>A,
            lda, <const float*>S, <const rocblas_float_complex*>U, ldu,
            <const rocblas_float_complex*>V, ldv, &lwork, <GesvdjInfo>params)
    check_status(status)
    return lwork

cpdef int zgesvdj_bufferSize(intptr_t handle, int jobz, int econ, int m, int n,
                             intptr_t A, int lda, intptr_t S, intptr_t U,
                             int ldu, intptr_t V, int ldv, intptr_t params):
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgesvdj_bufferSize(
            <Handle>handle, <EigMode>jobz, econ, m, n,
            <const rocblas_double_complex*>A, lda, <const double*>S,
            <const rocblas_double_complex*>U, ldu, <const rocblas_double_complex*>V,
            ldv, &lwork, <GesvdjInfo>params)
    check_status(status)
    return lwork

cpdef sgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgesvdj(<Handle>handle, <EigMode>jobz, econ, m, n,
                                   <float*>A, lda, <float*>S, <float*>U, ldu,
                                   <float*>V, ldv, <float*>work, lwork,
                                   <int*>info, <GesvdjInfo>params)
    check_status(status)

cpdef dgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgesvdj(<Handle>handle, <EigMode>jobz, econ, m, n,
                                   <double*>A, lda, <double*>S, <double*>U,
                                   ldu, <double*>V, ldv, <double*>work, lwork,
                                   <int*>info, <GesvdjInfo>params)
    check_status(status)

cpdef cgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgesvdj(
            <Handle>handle, <EigMode>jobz, econ, m, n, <rocblas_float_complex*>A, lda,
            <float*>S, <rocblas_float_complex*>U, ldu, <rocblas_float_complex*>V, ldv,
            <rocblas_float_complex*>work, lwork, <int*>info, <GesvdjInfo>params)
    check_status(status)

cpdef zgesvdj(intptr_t handle, int jobz, int econ, int m, int n, intptr_t A,
              int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
              intptr_t work, int lwork, intptr_t info, intptr_t params):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgesvdj(
            <Handle>handle, <EigMode>jobz, econ, m, n, <rocblas_double_complex*>A,
            lda, <double*>S, <rocblas_double_complex*>U, ldu, <rocblas_double_complex*>V,
            ldv, <rocblas_double_complex*>work, lwork, <int*>info, <GesvdjInfo>params)
    check_status(status)

cpdef int sgesvdjBatched_bufferSize(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t params, int batchSize) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgesvdjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, m, n, <float*>A, lda,
            <float*>S, <float*>U, ldu, <float*>V, ldv, &lwork,
            <GesvdjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int dgesvdjBatched_bufferSize(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t params, int batchSize) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgesvdjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, m, n, <double*>A, lda,
            <double*>S, <double*>U, ldu, <double*>V, ldv, &lwork,
            <GesvdjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int cgesvdjBatched_bufferSize(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t params, int batchSize) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgesvdjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, m, n, <rocblas_float_complex*>A, lda,
            <float*>S, <rocblas_float_complex*>U, ldu, <rocblas_float_complex*>V, ldv, &lwork,
            <GesvdjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int zgesvdjBatched_bufferSize(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t params, int batchSize) except? -1:
    cdef int lwork
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgesvdjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, m, n, <rocblas_double_complex*>A, lda,
            <double*>S, <rocblas_double_complex*>U, ldu, <rocblas_double_complex*>V, ldv,
            &lwork,
            <GesvdjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef sgesvdjBatched(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t work, int lwork, intptr_t info,
        intptr_t params, int batchSize):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgesvdjBatched(
            <Handle>handle, <EigMode>jobz, m, n, <float*>A, lda,
            <float*>S, <float*>U, ldu, <float*>V, ldv,
            <float*>work, lwork, <int*>info,
            <GesvdjInfo>params, batchSize)
    check_status(status)

cpdef dgesvdjBatched(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t work, int lwork, intptr_t info,
        intptr_t params, int batchSize):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgesvdjBatched(
            <Handle>handle, <EigMode>jobz, m, n, <double*>A, lda,
            <double*>S, <double*>U, ldu, <double*>V, ldv,
            <double*>work, lwork, <int*>info,
            <GesvdjInfo>params, batchSize)
    check_status(status)

cpdef cgesvdjBatched(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t work, int lwork, intptr_t info,
        intptr_t params, int batchSize):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgesvdjBatched(
            <Handle>handle, <EigMode>jobz, m, n, <rocblas_float_complex*>A, lda,
            <float*>S, <rocblas_float_complex*>U, ldu, <rocblas_float_complex*>V, ldv,
            <rocblas_float_complex*>work, lwork, <int*>info,
            <GesvdjInfo>params, batchSize)
    check_status(status)

cpdef zgesvdjBatched(
        intptr_t handle, int jobz, int m, int n, intptr_t A,
        int lda, intptr_t S, intptr_t U, int ldu, intptr_t V, int ldv,
        intptr_t work, int lwork, intptr_t info,
        intptr_t params, int batchSize):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgesvdjBatched(
            <Handle>handle, <EigMode>jobz, m, n, <rocblas_double_complex*>A, lda,
            <double*>S, <rocblas_double_complex*>U, ldu, <rocblas_double_complex*>V, ldv,
            <rocblas_double_complex*>work, lwork, <int*>info,
            <GesvdjInfo>params, batchSize)
    check_status(status)

# gesvda ... Approximate singular value decomposition
cpdef int sgesvdaStridedBatched_bufferSize(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, int batchSize):
    cdef int lwork
    status = hipsolverDnSgesvdaStridedBatched_bufferSize(
        <Handle>handle, <EigMode>jobz, rank, m, n, <const float*>d_A, lda,
        strideA, <const float*>d_S, strideS, <const float*>d_U, ldu, strideU,
        <const float*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int dgesvdaStridedBatched_bufferSize(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, int batchSize):
    cdef int lwork
    status = hipsolverDnDgesvdaStridedBatched_bufferSize(
        <Handle>handle, <EigMode>jobz, rank, m, n, <const double*>d_A, lda,
        strideA, <const double*>d_S, strideS, <const double*>d_U, ldu, strideU,
        <const double*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int cgesvdaStridedBatched_bufferSize(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, int batchSize):
    cdef int lwork
    status = hipsolverDnCgesvdaStridedBatched_bufferSize(
        <Handle>handle, <EigMode>jobz, rank, m, n, <const rocblas_float_complex*>d_A, lda,
        strideA, <const float*>d_S, strideS, <const rocblas_float_complex*>d_U, ldu,
        strideU, <const rocblas_float_complex*>d_V, ldv, strideV, &lwork, batchSize)
    check_status(status)
    return lwork

cpdef int zgesvdaStridedBatched_bufferSize(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, int batchSize):
    cdef int lwork
    status = hipsolverDnZgesvdaStridedBatched_bufferSize(
        <Handle>handle, <EigMode>jobz, rank, m, n, <const rocblas_double_complex*>d_A,
        lda, strideA, <const double*>d_S, strideS, <const rocblas_double_complex*>d_U,
        ldu, strideU, <const rocblas_double_complex*>d_V, ldv, strideV, &lwork,
        batchSize)
    check_status(status)
    return lwork

cpdef sgesvdaStridedBatched(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
        intptr_t h_R_nrmF, int batchSize):
    _setStream(handle)
    with nogil:
        status = hipsolverDnSgesvdaStridedBatched(
            <Handle>handle, <EigMode>jobz, rank, m, n, <const float*>d_A, lda,
            strideA, <float*>d_S, strideS, <float*>d_U, ldu, strideU,
            <float*>d_V, ldv, strideV, <float*>d_work, lwork, <int*>d_info,
            <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef dgesvdaStridedBatched(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
        intptr_t h_R_nrmF, int batchSize):
    _setStream(handle)
    with nogil:
        status = hipsolverDnDgesvdaStridedBatched(
            <Handle>handle, <EigMode>jobz, rank, m, n, <const double*>d_A, lda,
            strideA, <double*>d_S, strideS, <double*>d_U, ldu, strideU,
            <double*>d_V, ldv, strideV, <double*>d_work, lwork, <int*>d_info,
            <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef cgesvdaStridedBatched(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
        intptr_t h_R_nrmF, int batchSize):
    _setStream(handle)
    with nogil:
        status = hipsolverDnCgesvdaStridedBatched(
            <Handle>handle, <EigMode>jobz, rank, m, n, <const rocblas_float_complex*>d_A,
            lda, strideA, <float*>d_S, strideS, <rocblas_float_complex*>d_U, ldu, strideU,
            <rocblas_float_complex*>d_V, ldv, strideV, <rocblas_float_complex*>d_work, lwork,
            <int*>d_info, <double*>h_R_nrmF, batchSize)
    check_status(status)

cpdef zgesvdaStridedBatched(
        intptr_t handle, int jobz, int rank, int m, int n, intptr_t d_A,
        int lda, long long int strideA, intptr_t d_S, long long int strideS,
        intptr_t d_U, int ldu, long long int strideU, intptr_t d_V, int ldv,
        long long int strideV, intptr_t d_work, int lwork, intptr_t d_info,
        intptr_t h_R_nrmF, int batchSize):
    _setStream(handle)
    with nogil:
        status = hipsolverDnZgesvdaStridedBatched(
            <Handle>handle, <EigMode>jobz, rank, m, n,
            <const rocblas_double_complex*>d_A, lda, strideA, <double*>d_S, strideS,
            <rocblas_double_complex*>d_U, ldu, strideU, <rocblas_double_complex*>d_V, ldv,
            strideV, <rocblas_double_complex*>d_work, lwork, <int*>d_info,
            <double*>h_R_nrmF, batchSize)
    check_status(status)

# Standard symmetric eigenvalue solver
cpdef int ssyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = hipsolverDnSsyevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const float*>A,
            lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int dsyevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = hipsolverDnDsyevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const double*>A,
            lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef int cheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = hipsolverDnCheevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const rocblas_float_complex*>A,
            lda, <const float*>W, &lwork)
    check_status(status)
    return lwork

cpdef int zheevd_bufferSize(intptr_t handle, int jobz, int uplo, int n,
                            size_t A, int lda, size_t W) except? -1:
    cdef int lwork, status
    _setStream(handle)
    with nogil:
        status = hipsolverDnZheevd_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const rocblas_double_complex*>A,
            lda, <const double*>W, &lwork)
    check_status(status)
    return lwork

cpdef ssyevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    _setStream(handle)
    with nogil:
        status = hipsolverDnSsyevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <float*>A, lda, <float*>W,
            <float*>work, lwork, <int*>info)
    check_status(status)

cpdef dsyevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    _setStream(handle)
    with nogil:
        status = hipsolverDnDsyevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <double*>A, lda, <double*>W,
            <double*>work, lwork, <int*>info)
    check_status(status)

cpdef cheevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    _setStream(handle)
    with nogil:
        status = hipsolverDnCheevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <rocblas_float_complex*>A, lda, <float*>W,
            <rocblas_float_complex*>work, lwork, <int*>info)
    check_status(status)

cpdef zheevd(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info):
    cdef int status
    _setStream(handle)
    with nogil:
        status = hipsolverDnZheevd(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <rocblas_double_complex*>A, lda, <double*>W,
            <rocblas_double_complex*>work, lwork, <int*>info)
    check_status(status)

# Symmetric eigenvalue solver via Jacobi method
cpdef intptr_t createSyevjInfo() except? 0:
    cdef SyevjInfo info
    status = hipsolverDnCreateSyevjInfo(&info)
    check_status(status)
    return <intptr_t>info

cpdef destroySyevjInfo(intptr_t info):
    status = hipsolverDnDestroySyevjInfo(<SyevjInfo>info)
    check_status(status)

cpdef xsyevjSetTolerance(intptr_t info, double tolerance):
    status = cusolverDnXsyevjSetTolerance(<SyevjInfo>info, tolerance)
    check_status(status)

cpdef xsyevjSetMaxSweeps(intptr_t info, int max_sweeps):
    status = cusolverDnXsyevjSetMaxSweeps(<SyevjInfo>info, max_sweeps)
    check_status(status)

cpdef xsyevjSetSortEig(intptr_t info, int sort_eig):
    status = hipsolverDnXsyevjSetSortEig(<SyevjInfo>info, sort_eig)
    check_status(status)

cpdef double xsyevjGetResidual(intptr_t handle, intptr_t info):
    cdef double residual
    status = cusolverDnXsyevjGetResidual(
        <Handle>handle, <SyevjInfo>info, &residual)
    check_status(status)
    return residual

cpdef int xsyevjGetSweeps(intptr_t handle, intptr_t info):
    cdef int executed_sweeps
    status = cusolverDnXsyevjGetSweeps(
        <Handle>handle, <SyevjInfo>info, &executed_sweeps)
    check_status(status)
    return executed_sweeps

cpdef int ssyevj_bufferSize(intptr_t handle, int jobz, int uplo,
                            int n, size_t A, int lda, size_t W,
                            intptr_t params) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnSsyevj_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const float*>A,
            lda, <const float*>W, &lwork, <SyevjInfo>params)
    check_status(status)
    return lwork

cpdef int dsyevj_bufferSize(intptr_t handle, int jobz, int uplo,
                            int n, size_t A, int lda, size_t W,
                            intptr_t params) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnDsyevj_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const double*>A,
            lda, <const double*>W, &lwork, <SyevjInfo>params)
    check_status(status)
    return lwork

cpdef int cheevj_bufferSize(intptr_t handle, int jobz, int uplo,
                            int n, size_t A, int lda, size_t W,
                            intptr_t params) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnCheevj_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const rocblas_float_complex*>A,
            lda, <const float*>W, &lwork, <SyevjInfo>params)
    check_status(status)
    return lwork

cpdef int zheevj_bufferSize(intptr_t handle, int jobz, int uplo,
                            int n, size_t A, int lda, size_t W,
                            intptr_t params) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnZheevj_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const rocblas_double_complex*>A,
            lda, <const double*>W, &lwork, <SyevjInfo>params)
    check_status(status)
    return lwork

cpdef ssyevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnSsyevj(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <float*>A, lda, <float*>W,
            <float*>work, lwork, <int*>info, <SyevjInfo>params)
    check_status(status)

cpdef dsyevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnDsyevj(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <double*>A, lda, <double*>W,
            <double*>work, lwork, <int*>info, <SyevjInfo>params)
    check_status(status)

cpdef cheevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnCheevj(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <rocblas_float_complex*>A, lda, <float*>W,
            <rocblas_float_complex*>work, lwork, <int*>info, <SyevjInfo>params)
    check_status(status)

cpdef zheevj(intptr_t handle, int jobz, int uplo, int n, size_t A, int lda,
             size_t W, size_t work, int lwork, size_t info, intptr_t params):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnZheevj(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <rocblas_double_complex*>A, lda, <double*>W,
            <rocblas_double_complex*>work, lwork, <int*>info, <SyevjInfo>params)
    check_status(status)

# Batched symmetric eigenvalue solver via Jacobi method

cpdef int ssyevjBatched_bufferSize(
        intptr_t handle, int jobz, int uplo, int n,
        size_t A, int lda, size_t W, intptr_t params,
        int batchSize) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnSsyevjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const float *>A, lda, <const float *>W, &lwork,
            <SyevjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int dsyevjBatched_bufferSize(
        intptr_t handle, int jobz, int uplo, int n,
        size_t A, int lda, size_t W, intptr_t params,
        int batchSize) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnDsyevjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const double *>A, lda, <const double *>W, &lwork,
            <SyevjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int cheevjBatched_bufferSize(
        intptr_t handle, int jobz, int uplo, int n,
        size_t A, int lda, size_t W, intptr_t params,
        int batchSize) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnCheevjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const rocblas_float_complex *>A, lda, <const float *>W, &lwork,
            <SyevjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef int zheevjBatched_bufferSize(
        intptr_t handle, int jobz, int uplo, int n,
        size_t A, int lda, size_t W, intptr_t params,
        int batchSize) except? -1:
    cdef int lwork, status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnZheevjBatched_bufferSize(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <const rocblas_double_complex *>A, lda, <const double *>W, &lwork,
            <SyevjInfo>params, batchSize)
    check_status(status)
    return lwork

cpdef ssyevjBatched(intptr_t handle, int jobz, int uplo, int n,
                    size_t A, int lda, size_t W, size_t work, int lwork,
                    size_t info, intptr_t params, int batchSize):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnSsyevjBatched(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <float*>A, lda, <float*>W,
            <float*>work, lwork, <int*>info, <SyevjInfo>params, batchSize)
    check_status(status)

cpdef dsyevjBatched(intptr_t handle, int jobz, int uplo, int n,
                    size_t A, int lda, size_t W, size_t work, int lwork,
                    size_t info, intptr_t params, int batchSize):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnDsyevjBatched(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <double*>A, lda, <double*>W,
            <double*>work, lwork, <int*>info, <SyevjInfo>params, batchSize)
    check_status(status)

cpdef cheevjBatched(intptr_t handle, int jobz, int uplo, int n,
                    size_t A, int lda, size_t W, size_t work, int lwork,
                    size_t info, intptr_t params, int batchSize):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnCheevjBatched(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <rocblas_float_complex*>A, lda, <float*>W,
            <rocblas_float_complex*>work, lwork, <int*>info, <SyevjInfo>params, batchSize)
    check_status(status)

cpdef zheevjBatched(intptr_t handle, int jobz, int uplo, int n,
                    size_t A, int lda, size_t W, size_t work, int lwork,
                    size_t info, intptr_t params, int batchSize):
    cdef int status
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnZheevjBatched(
            <Handle>handle, <EigMode>jobz, <FillMode>uplo, n,
            <rocblas_double_complex*>A, lda, <double*>W,
            <rocblas_double_complex*>work, lwork, <int*>info,
            <SyevjInfo>params, batchSize)
    check_status(status)

# dense eigenvalue solver (64bit)
cpdef (size_t, size_t) xsyevd_bufferSize(  # noqa
        intptr_t handle, intptr_t params, int jobz, int uplo,
        int64_t n, int dataTypeA, intptr_t A, int64_t lda,
        int dataTypeW, intptr_t W, int computeType) except *:
    cdef size_t workspaceInBytesOnDevice, workspaceInBytesOnHost
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnXsyevd_bufferSize(
            <Handle>handle, <Params>params, <EigMode> jobz, <FillMode> uplo, n,
            <DataType>dataTypeA, <void*>A, lda,
            <DataType>dataTypeW, <void*>W, <DataType>computeType,
            &workspaceInBytesOnDevice, &workspaceInBytesOnHost)
    check_status(status)
    return workspaceInBytesOnDevice, workspaceInBytesOnHost

cpdef xsyevd(
        intptr_t handle, intptr_t params, int jobz, int uplo,
        int64_t n, int dataTypeA, intptr_t A, int64_t lda,
        int dataTypeW, intptr_t W, int computeType, intptr_t bufferOnDevice,
        size_t workspaceInBytesOnDevice, intptr_t bufferOnHost,
        size_t workspaceInBytesOnHost, intptr_t info):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = hipsolverDnXsyevd(
            <Handle>handle, <Params>params, <EigMode>jobz, <FillMode>uplo, n,
            <DataType>dataTypeA, <void*>A, lda,
            <DataType>dataTypeW, <void*>W, <DataType>computeType,
            <void*>bufferOnDevice, workspaceInBytesOnDevice,
            <void*>bufferOnHost, workspaceInBytesOnHost, <int*>info)
    check_status(status)


###############################################################################
# Sparse LAPACK Functions
###############################################################################
cpdef scsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                  size_t b, float tol, int reorder, size_t x,
                  size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpScsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const float*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const float*> b,
            tol, reorder, <float*> x, <int*> singularity)
    check_status(status)

cpdef dcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                  size_t b, double tol, int reorder, size_t x,
                  size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpDcsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const double*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const double*> b,
            tol, reorder, <double*> x, <int*> singularity)
    check_status(status)

cpdef ccsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrVal, size_t csrRowPtr, size_t csrColInd, size_t b,
                  float tol, int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpCcsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const rocblas_float_complex*>csrVal, <const int*>csrRowPtr,
            <const int*>csrColInd, <const rocblas_float_complex*>b, tol, reorder,
            <rocblas_float_complex*>x, <int*>singularity)
    check_status(status)

cpdef zcsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA,
                  size_t csrVal, size_t csrRowPtr, size_t csrColInd, size_t b,
                  double tol, int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpZcsrlsvchol(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const rocblas_double_complex*>csrVal, <const int*>csrRowPtr,
            <const int*>csrColInd, <const rocblas_double_complex*>b, tol, reorder,
            <rocblas_double_complex*>x, <int*>singularity)
    check_status(status)

cpdef scsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, float tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpScsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const float*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const float*> b,
            tol, reorder, <float*> x, <int*> singularity)
    check_status(status)

cpdef dcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrValA,
                size_t csrRowPtrA, size_t csrColIndA, size_t b, double tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpDcsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr> descrA,
            <const double*> csrValA, <const int*> csrRowPtrA,
            <const int*> csrColIndA, <const double*> b,
            tol, reorder, <double*> x, <int*> singularity)
    check_status(status)

cpdef ccsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                size_t csrRowPtr, size_t csrColInd, size_t b, float tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpCcsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const rocblas_float_complex*>csrVal, <const int*>csrRowPtr,
            <const int*>csrColInd, <const rocblas_float_complex*>b, tol, reorder,
            <rocblas_float_complex*>x, <int*>singularity)
    check_status(status)

cpdef zcsrlsvqr(intptr_t handle, int m, int nnz, size_t descrA, size_t csrVal,
                size_t csrRowPtr, size_t csrColInd, size_t b, double tol,
                int reorder, size_t x, size_t singularity):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpZcsrlsvqr(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const rocblas_double_complex*>csrVal, <const int*>csrRowPtr,
            <const int*>csrColInd, <const rocblas_double_complex*>b, tol, reorder,
            <rocblas_double_complex*>x, <int*>singularity)
    check_status(status)

cpdef scsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 float mu0, size_t x0, int maxite, float eps, size_t mu,
                 size_t x):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpScsreigvsi(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const float*>csrValA, <const int*>csrRowPtrA,
            <const int*>csrColIndA, mu0, <const float*>x0, maxite, eps,
            <float*>mu, <float*>x)
    check_status(status)

cpdef dcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 double mu0, size_t x0, int maxite, double eps, size_t mu,
                 size_t x):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpDcsreigvsi(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const double*>csrValA, <const int*>csrRowPtrA,
            <const int*>csrColIndA, mu0, <const double*>x0, maxite, eps,
            <double*>mu, <double*>x)
    check_status(status)

cpdef ccsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 size_t mu0, size_t x0, int maxite, float eps, size_t mu,
                 size_t x):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpCcsreigvsi(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const rocblas_float_complex*>csrValA, <const int*>csrRowPtrA,
            <const int*>csrColIndA, (<rocblas_float_complex*>mu0)[0], <const rocblas_float_complex*>x0,
            maxite, eps, <rocblas_float_complex*>mu, <rocblas_float_complex*>x)
    check_status(status)

cpdef zcsreigvsi(intptr_t handle, int m, int nnz, size_t descrA,
                 size_t csrValA, size_t csrRowPtrA, size_t csrColIndA,
                 size_t mu0, size_t x0, int maxite, double eps, size_t mu,
                 size_t x):
    cdef int status
    _spSetStream(handle)
    with nogil:
        status = cusolverSpZcsreigvsi(
            <SpHandle>handle, m, nnz, <const MatDescr>descrA,
            <const rocblas_double_complex*>csrValA, <const int*>csrRowPtrA,
            <const int*>csrColIndA, (<rocblas_double_complex*>mu0)[0],
            <const rocblas_double_complex*>x0, maxite,
            eps, <rocblas_double_complex*>mu, <rocblas_double_complex*>x)
    check_status(status)
