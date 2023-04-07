#ifndef INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H
#define INCLUDE_GUARD_HIP_CUPY_ROCScublasOperation_t opOLVER_H

#include "cupy_hip.h"
#include "cupy_hipblas.h"
#include <stdexcept>  // for gcc 10.0


extern "C" {
// TODO(leofang): perhaps these should be merged with the support of hipBLAS?
static rocblas_fill convert_rocblas_fill(cublasFillMode_t mode) {
    switch(static_cast<int>(mode)) {
        case 0 /* CUBLAS_FILL_MODE_LOWER */: return rocblas_fill_lower;
        case 1 /* CUBLAS_FILL_MODE_UPPER */: return rocblas_fill_upper;
        default: throw std::runtime_error("unrecognized mode");
    }
}

static rocblas_operation convert_rocblas_operation(cublasOperation_t op) {
    return static_cast<rocblas_operation>(static_cast<int>(op) + 111);
}

static rocblas_side convert_rocblas_side(cublasSideMode_t mode) {
    return static_cast<rocblas_side>(static_cast<int>(mode) + 141);
}

#if HIP_VERSION >= 309
static rocblas_svect convert_rocblas_svect(signed char mode) {
    switch(mode) {
        case 'A': return rocblas_svect_all;
        case 'S': return rocblas_svect_singular;
        case 'O': return rocblas_svect_overwrite;
        case 'N': return rocblas_svect_none;
        default: throw std::runtime_error("unrecognized mode");
    }
}
#endif

#if HIP_VERSION >= 540
static hipsolverFillMode_t convert_to_hipsolverFill(cublasFillMode_t mode) {
    switch(static_cast<int>(mode)) {
        case 0 /* CUBLAS_FILL_MODE_LOWER */: return HIPSOLVER_FILL_MODE_LOWER;
        case 1 /* CUBLAS_FILL_MODE_UPPER */: return HIPSOLVER_FILL_MODE_UPPER;
        default: throw std::runtime_error("unrecognized mode");
    }
}

static hipsolverOperation_t convert_hipsolver_operation(cublasOperation_t op) {
    return static_cast<hipsolverOperation_t>(static_cast<int>(op) + 111);
}

static hipsolverSideMode_t convert_hipsolver_side(cublasSideMode_t mode) {
    return static_cast<hipsolverSideMode_t>(static_cast<int>(mode) + 141);
}

#endif
// rocSOLVER
/* ---------- helpers ---------- */

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) {
  #if HIP_VERSION >= 540
    return hipsolverCreate(handle);
  #else
    return rocblas_create_handle(handle);
  #endif
}

cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
  #if HIP_VERSION >= 540
    return hipsolverDestroy(handle);
  #else
    return rocblas_destroy_handle(handle);
  #endif
}

cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle,
                                     cudaStream_t *streamId) {
  #if HIP_VERSION >= 540
    return hipsolverGetStream(handle, streamId);
  #else
    return rocblas_get_stream(handle, streamId);
  #endif
}

cusolverStatus_t cusolverDnSetStream (cusolverDnHandle_t handle,
                                      cudaStream_t streamId) {
  #if HIP_VERSION >= 540
    return hipsolverSetStream(handle, streamId);
  #else
    return rocblas_set_stream(handle, streamId);
  #endif
}

cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* val) {
    switch(type) {
        case MAJOR_VERSION: { *val = ROCSOLVER_VERSION_MAJOR; break; }
        case MINOR_VERSION: { *val = ROCSOLVER_VERSION_MINOR; break; }
        case PATCH_LEVEL:   { *val = ROCSOLVER_VERSION_PATCH; break; }
        default: throw std::runtime_error("invalid type");
    }
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_SUCCESS;
  #else
    return rocblas_status_success;
  #endif
}


typedef enum cusolverDnParams_t {};

cusolverStatus_t cusolverDnCreateParams(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDestroyParams(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

/* ---------- potrf ---------- */
cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             float *A,
                                             int lda,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverSpotrf_bufferSize(handle, convert_to_hipsolverFill(uplo), n, A, lda, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             double *A,
                                             int lda,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDpotrf_bufferSize(handle, convert_to_hipsolverFill(uplo), n, A, lda, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             cuComplex *A,
                                             int lda,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverCpotrf_bufferSize(handle, convert_to_hipsolverFill(uplo), n,
                        reinterpret_cast<hipFloatComplex*>(A), lda, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             cuDoubleComplex *A,
                                             int lda,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZpotrf_bufferSize(handle, convert_to_hipsolverFill(uplo), n,
                        reinterpret_cast<hipDoubleComplex*>(A), lda, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *Workspace,
                                  int Lwork,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverSpotrf(handle, convert_to_hipsolverFill(uplo), n,
                        A, lda, Workspace, Lwork, devInfo);
  #else
    // ignore Workspace and Lwork as rocSOLVER does not need them
    return rocsolver_spotrf(handle, convert_rocblas_fill(uplo),
                            n, A, lda, devInfo);
  #endif
}

cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *Workspace,
                                  int Lwork,
                                  int *devInfo ) {
  #if HIP_VERSION >= 540
    return hipsolverDpotrf(handle, convert_to_hipsolverFill(uplo), n,
                        A, lda, Workspace, Lwork, devInfo);
  #else
    // ignore Workspace and Lwork as rocSOLVER does not need them
    return rocsolver_dpotrf(handle, convert_rocblas_fill(uplo),
                            n, A, lda, devInfo);
  #endif
}

cusolverStatus_t cusolverDnCpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  cuComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 504
    // ignore Workspace and Lwork as rocSOLVER does not need them
    return rocsolver_cpotrf(handle, convert_rocblas_fill(uplo), n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda, devInfo);
    #else
    return hipsolverCpotrf(handle, convert_to_hipsolverFill(uplo), n,
                            reinterpret_cast<hipFloatComplex*>(A), lda, reinterpret_cast<hipFloatComplex*>(Workspace), Lwork, devInfo);
    #endif
}

cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  cuDoubleComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 504
    // ignore Workspace and Lwork as rocSOLVER does not need them
    return rocsolver_zpotrf(handle, convert_rocblas_fill(uplo), n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda, devInfo);
    #else
    return hipsolverZpotrf(handle, convert_to_hipsolverFill(uplo), n,
                            reinterpret_cast<hipDoubleComplex*>(A), lda, reinterpret_cast<hipDoubleComplex*>(Workspace), Lwork, devInfo);
    #endif
}

cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo,
                                         int n,
                                         float *Aarray[],
                                         int lda,
                                         int *infoArray,
                                         int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverSpotrfBatched(handle, convert_to_hipsolverFill(uplo), n,
                                    Aarray, lda, nullptr, 0, infoArray, batchSize);
  #else
    return rocsolver_spotrf_batched(handle, convert_rocblas_fill(uplo),
                                    n, Aarray, lda, infoArray, batchSize);
  #endif
}

cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo,
                                         int n,
                                         double *Aarray[],
                                         int lda,
                                         int *infoArray,
                                         int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverDpotrfBatched(handle, convert_to_hipsolverFill(uplo), n,
                                    Aarray, lda, nullptr, 0, infoArray, batchSize);
  #else
    return rocsolver_dpotrf_batched(handle, convert_rocblas_fill(uplo),
                                    n, Aarray, lda, infoArray, batchSize);
  #endif
}

cusolverStatus_t cusolverDnCpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo,
                                         int n,
                                         cuComplex *Aarray[],
                                         int lda,
                                         int *infoArray,
                                         int batchSize) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_cpotrf_batched(handle, convert_rocblas_fill(uplo), n,
                                    reinterpret_cast<rocblas_float_complex* const*>(Aarray), lda,
                                    infoArray, batchSize);
    #else
    return hipsolverCpotrfBatched(handle, convert_to_hipsolverFill(uplo), n,
                                    reinterpret_cast<hipFloatComplex**>(Aarray), lda, nullptr, 0,
                                    infoArray, batchSize);
    #endif
}

cusolverStatus_t cusolverDnZpotrfBatched(cusolverDnHandle_t handle,
                                         cublasFillMode_t uplo,
                                         int n,
                                         cuDoubleComplex *Aarray[],
                                         int lda,
                                         int *infoArray,
                                         int batchSize) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_zpotrf_batched(handle, convert_rocblas_fill(uplo), n,
                                    reinterpret_cast<rocblas_double_complex* const*>(Aarray), lda,
                                    infoArray, batchSize);
    #else
    return hipsolverZpotrfBatched(handle, convert_to_hipsolverFill(uplo), n,
                                    reinterpret_cast<hipDoubleComplex**>(Aarray), lda, nullptr, 0,
                                    infoArray, batchSize);
    #endif
}


/* ---------- getrf ---------- */
cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             float *A,
                                             int lda,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             double *A,
                                             int lda,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             cuComplex *A,
                                             int lda,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverCgetrf_bufferSize(handle, m, n, reinterpret_cast<hipFloatComplex*>(A), lda, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             cuDoubleComplex *A,
                                             int lda,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZgetrf_bufferSize(handle, m, n, reinterpret_cast<hipDoubleComplex*>(A), lda, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *Workspace,
                                  int *devIpiv,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverSgetrf(handle, m, n, A, lda, Workspace, 0, devIpiv, devInfo);
  #else
    // ignore Workspace as rocSOLVER does not need it
    return rocsolver_sgetrf(handle, m, n, A, lda, devIpiv, devInfo);
  #endif
}

cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *Workspace,
                                  int *devIpiv,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverDgetrf(handle, m, n, A, lda, Workspace, 0, devIpiv, devInfo);
  #else
    // ignore Workspace as rocSOLVER does not need it
    return rocsolver_dgetrf(handle, m, n, A, lda, devIpiv, devInfo);
  #endif
}

cusolverStatus_t cusolverDnCgetrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  cuComplex *Workspace,
                                  int *devIpiv,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverCgetrf(handle, m, n, reinterpret_cast<hipFloatComplex*>(A), lda,
                        reinterpret_cast<hipFloatComplex*>(Workspace), 0, devIpiv, devInfo);
  #else
    // ignore Workspace as rocSOLVER does not need it
    return rocsolver_cgetrf(handle, m, n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            devIpiv, devInfo);
  #endif
}

cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  cuDoubleComplex *Workspace,
                                  int *devIpiv,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverZgetrf(handle, m, n, reinterpret_cast<hipDoubleComplex*>(A), lda,
                        reinterpret_cast<hipDoubleComplex*>(Workspace), 0, devIpiv, devInfo);
  #else
    // ignore Workspace as rocSOLVER does not need it
    return rocsolver_zgetrf(handle, m, n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            devIpiv, devInfo);
  #endif
}


/* ---------- getrs ---------- */
cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const float *A,
                                  int lda,
                                  const int *devIpiv,
                                  float *B,
                                  int ldb,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverSgetrs(handle, convert_hipsolver_operation(trans),
                            n, nrhs, const_cast<float*>(A), lda,
                            const_cast<int*>(devIpiv), B, ldb, nullptr, 0, devInfo);
  #else
    // ignore devInfo as rocSOLVER does not need it
    return rocsolver_sgetrs(handle,
                            convert_rocblas_operation(trans),
                            n, nrhs, const_cast<float*>(A), lda, devIpiv, B, ldb);
  #endif
}

cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const double *A,
                                  int lda,
                                  const int *devIpiv,
                                  double *B,
                                  int ldb,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverDgetrs(handle, convert_hipsolver_operation(trans),
                            n, nrhs, const_cast<double*>(A), lda,
                            const_cast<int*>(devIpiv), B, ldb, nullptr, 0, devInfo);
  #else
    // ignore devInfo as rocSOLVER does not need it
    return rocsolver_dgetrs(handle,
                            convert_rocblas_operation(trans),
                            n, nrhs, const_cast<double*>(A), lda, devIpiv, B, ldb);
  #endif
}

cusolverStatus_t cusolverDnCgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const cuComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  cuComplex *B,
                                  int ldb,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverCgetrs(handle, convert_hipsolver_operation(trans),
                            n, nrhs, reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(A)),
                            lda, const_cast<int*>(devIpiv),
                            reinterpret_cast<hipFloatComplex*>(B), ldb, nullptr, 0, devInfo);
  #else
    // ignore devInfo as rocSOLVER does not need it
    return rocsolver_cgetrs(handle,
                            convert_rocblas_operation(trans),
                            n, nrhs,
                            (rocblas_float_complex*)(A), lda,
                            devIpiv,
                            reinterpret_cast<rocblas_float_complex*>(B), ldb);
  #endif
}

cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n,
                                  int nrhs,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const int *devIpiv,
                                  cuDoubleComplex *B,
                                  int ldb,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverZgetrs(handle, convert_hipsolver_operation(trans),
                            n, nrhs, reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(A)),
                            lda, const_cast<int*>(devIpiv),
                            reinterpret_cast<hipDoubleComplex*>(B), ldb, nullptr, 0, devInfo);
  #else
    // ignore devInfo as rocSOLVER does not need it
    return rocsolver_zgetrs(handle,
                            convert_rocblas_operation(trans),
                            n, nrhs,
                            (rocblas_double_complex*)(A), lda,
                            devIpiv,
                            reinterpret_cast<rocblas_double_complex*>(B), ldb);
  #endif
}


/* ---------- geqrf ---------- */
cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             float *A,
                                             int lda,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             double *A,
                                             int lda,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             cuComplex *A,
                                             int lda,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverCgeqrf_bufferSize(handle, m, n, reinterpret_cast<hipFloatComplex*>(A), lda, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             cuDoubleComplex *A,
                                             int lda,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZgeqrf_bufferSize(handle, m, n, reinterpret_cast<hipDoubleComplex*>(A), lda, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *TAU,
                                  float *Workspace,
                                  int Lwork,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
  #else
    // ignore Workspace, Lwork and devInfo as rocSOLVER does not need them
    return rocsolver_sgeqrf(handle, m, n, A, lda, TAU);
  #endif
}

cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *TAU,
                                  double *Workspace,
                                  int Lwork,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
  #else
    // ignore Workspace, Lwork and devInfo as rocSOLVER does not need them
    return rocsolver_dgeqrf(handle, m, n, A, lda, TAU);
  #endif
}

cusolverStatus_t cusolverDnCgeqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  cuComplex *TAU,
                                  cuComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverCgeqrf(handle, m, n, reinterpret_cast<hipFloatComplex*>(A),
                            lda, reinterpret_cast<hipFloatComplex*>(TAU),
                            reinterpret_cast<hipFloatComplex*>(Workspace), Lwork, devInfo);
  #else
    // ignore Workspace, Lwork and devInfo as rocSOLVER does not need them
    return rocsolver_cgeqrf(handle, m, n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            reinterpret_cast<rocblas_float_complex*>(TAU));
  #endif
}

cusolverStatus_t cusolverDnZgeqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  cuDoubleComplex *TAU,
                                  cuDoubleComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverZgeqrf(handle, m, n, reinterpret_cast<hipDoubleComplex*>(A),
                            lda, reinterpret_cast<hipDoubleComplex*>(TAU),
                            reinterpret_cast<hipDoubleComplex*>(Workspace), Lwork, devInfo);
  #else
    // ignore Workspace, Lwork and devInfo as rocSOLVER does not need them
    return rocsolver_zgeqrf(handle, m, n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            reinterpret_cast<rocblas_double_complex*>(TAU));
  #endif
}


/* ---------- orgqr ---------- */
cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int k,
                                             const float *A,
                                             int lda,
                                             const float *tau,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverSorgqr_bufferSize(handle, m, n, k, const_cast<float*>(A), lda, const_cast<float*>(tau), lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int k,
                                             const double *A,
                                             int lda,
                                             const double *tau,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDorgqr_bufferSize(handle, m, n, k, const_cast<double*>(A), lda, const_cast<double*>(tau), lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSorgqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  float *A,
                                  int lda,
                                  const float *tau,
                                  float *work,
                                  int lwork,
                                  int *info) {
  #if HIP_VERSION >= 540
    return hipsolverSorgqr(handle, m, n, k, A, lda, const_cast<float*>(tau), work, lwork, info);
  #else
    // ignore work, lwork and info as rocSOLVER does not need them
    return rocsolver_sorgqr(handle, m, n, k, A, lda, const_cast<float*>(tau));
  #endif
}

cusolverStatus_t cusolverDnDorgqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  double *A,
                                  int lda,
                                  const double *tau,
                                  double *work,
                                  int lwork,
                                  int *info) {
  #if HIP_VERSION >= 540
    return hipsolverDorgqr(handle, m, n, k, A, lda, const_cast<double*>(tau), work, lwork, info);
  #else
    // ignore work, lwork and info as rocSOLVER does not need them
    return rocsolver_dorgqr(handle, m, n, k, A, lda, const_cast<double*>(tau));
  #endif
}


/* ---------- ungqr ---------- */
cusolverStatus_t cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int k,
                                             const cuComplex *A,
                                             int lda,
                                             const cuComplex *tau,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDnCungqr_bufferSize(handle, m, n, k,
                                        reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(A)),
                                        lda, reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(tau)), lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int k,
                                             const cuDoubleComplex *A,
                                             int lda,
                                             const cuDoubleComplex *tau,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZungqr_bufferSize(handle, m, n, k, reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(A)),
                                        lda, reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(tau)), lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCungqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  cuComplex *A,
                                  int lda,
                                  const cuComplex *tau,
                                  cuComplex *work,
                                  int lwork,
                                  int *info) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 504
    // ignore work, lwork and info as rocSOLVER does not need them
    return rocsolver_cungqr(handle, m, n, k,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            reinterpret_cast<rocblas_float_complex*>(const_cast<cuComplex*>(tau)));
    #else
    return hipsolverCungqr(handle, m, n, k,
                            reinterpret_cast<hipFloatComplex*>(A), lda,
                            reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(tau)),
                            reinterpret_cast<hipFloatComplex*>(work), lwork, info);
    #endif
}

cusolverStatus_t cusolverDnZungqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  cuDoubleComplex *A,
                                  int lda,
                                  const cuDoubleComplex *tau,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  int *info) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 504
    // ignore work, lwork and info as rocSOLVER does not need them
    return rocsolver_zungqr(handle, m, n, k,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            reinterpret_cast<rocblas_double_complex*>(const_cast<cuDoubleComplex*>(tau)));
    #else
    return hipsolverZungqr(handle, m, n, k, reinterpret_cast<hipDoubleComplex*>(A), lda,
                            reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(tau)),
                            reinterpret_cast<hipDoubleComplex*>(work), lwork, info);
    #endif
}


/* ---------- ormqr ---------- */
cusolverStatus_t cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans,
                                             int m,
                                             int n,
                                             int k,
                                             const float *A,
                                             int lda,
                                             const float *tau,
                                             const float *C,
                                             int ldc,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverSormqr_bufferSize(handle, convert_hipsolver_side(side),
                                        convert_hipsolver_operation(trans), m, n, k,
                                        const_cast<float*>(A), lda, const_cast<float*>(tau),
                                        const_cast<float*>(C), ldc, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans,
                                             int m,
                                             int n,
                                             int k,
                                             const double *A,
                                             int lda,
                                             const double *tau,
                                             const double *C,
                                             int ldc,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDormqr_bufferSize(handle, convert_hipsolver_side(side),
                                        convert_hipsolver_operation(trans), m, n, k,
                                        const_cast<double*>(A), lda, const_cast<double*>(tau),
                                        const_cast<double*>(C), ldc, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const float *A,
                                  int lda,
                                  const float *tau,
                                  float *C,
                                  int ldc,
                                  float *work,
                                  int lwork,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverSormqr(handle, convert_hipsolver_side(side), convert_hipsolver_operation(trans),
                            m, n, k, const_cast<float*>(A), lda, const_cast<float*>(tau),
                            C, ldc, work, lwork, devInfo);
  #else
    // ignore work, lwork and devInfo as rocSOLVER does not need them
    return rocsolver_sormqr(handle,
                            convert_rocblas_side(side),
                            convert_rocblas_operation(trans),
                            m, n, k,
                            const_cast<float*>(A), lda,
                            const_cast<float*>(tau),
                            C, ldc);
  #endif
}

cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const double *A,
                                  int lda,
                                  const double *tau,
                                  double *C,
                                  int ldc,
                                  double *work,
                                  int lwork,
                                  int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverDormqr(handle, convert_hipsolver_side(side),
                            convert_hipsolver_operation(trans),
                            m, n, k, const_cast<double*>(A), lda,
                            const_cast<double*>(tau), C, ldc, work, lwork, devInfo);
  #else
    // ignore work, lwork and devInfo as rocSOLVER does not need them
    return rocsolver_dormqr(handle,
                            convert_rocblas_side(side),
                            convert_rocblas_operation(trans),
                            m, n, k,
                            const_cast<double*>(A), lda,
                            const_cast<double*>(tau),
                            C, ldc);
  #endif
}


/* ---------- unmqr ---------- */
cusolverStatus_t cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans,
                                             int m,
                                             int n,
                                             int k,
                                             const cuComplex *A,
                                             int lda,
                                             const cuComplex *tau,
                                             const cuComplex *C,
                                             int ldc,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverCunmqr_bufferSize(handle, convert_hipsolver_side(side),
                                        convert_hipsolver_operation(trans), m, n, k,
                                        reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(A)), lda,
                                        reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(tau)),
                                        reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(C)), ldc, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZunmqr_bufferSize(cusolverDnHandle_t handle,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans,
                                             int m,
                                             int n,
                                             int k,
                                             const cuDoubleComplex *A,
                                             int lda,
                                             const cuDoubleComplex *tau,
                                             const cuDoubleComplex *C,
                                             int ldc,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZunmqr_bufferSize(handle, convert_hipsolver_side(side),
                                        convert_hipsolver_operation(trans), m, n, k,
                                        reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(A)), lda,
                                        reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(tau)),
                                        reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(C)), ldc, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCunmqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const cuComplex *A,
                                  int lda,
                                  const cuComplex *tau,
                                  cuComplex *C,
                                  int ldc,
                                  cuComplex *work,
                                  int lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work, lwork and devInfo as rocSOLVER does not need them
    return rocsolver_cunmqr(handle, convert_rocblas_side(side), convert_rocblas_operation(trans),
                            m, n, k, reinterpret_cast<rocblas_float_complex*>(const_cast<cuComplex*>(A)),
                            lda, reinterpret_cast<rocblas_float_complex*>(const_cast<cuComplex*>(tau)),
                            reinterpret_cast<rocblas_float_complex*>(C), ldc);
    #else
    return hipsolverCunmqr(handle, convert_hipsolver_side(side), convert_hipsolver_operation(trans),
                            m, n, k, reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(A)),
                            lda, reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(tau)),
                            reinterpret_cast<hipFloatComplex*>(C), ldc, reinterpret_cast<hipFloatComplex*>(work), lwork, devInfo);
    #endif
}

cusolverStatus_t cusolverDnZunmqr(cusolverDnHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  int k,
                                  const cuDoubleComplex *A,
                                  int lda,
                                  const cuDoubleComplex *tau,
                                  cuDoubleComplex *C,
                                  int ldc,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work, lwork and devInfo as rocSOLVER does not need them
    return rocsolver_zunmqr(handle, convert_rocblas_side(side), convert_rocblas_operation(trans),
                            m, n, k, reinterpret_cast<rocblas_double_complex*>(const_cast<cuDoubleComplex*>(A)),
                            lda, reinterpret_cast<rocblas_double_complex*>(const_cast<cuDoubleComplex*>(tau)),
                            reinterpret_cast<rocblas_double_complex*>(C), ldc);
    #else
    return hipsolverZunmqr(handle, convert_hipsolver_side(side), convert_hipsolver_operation(trans),
                            m, n, k, reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(A)),
                            lda, reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(tau)),
                            reinterpret_cast<hipDoubleComplex*>(C), ldc, reinterpret_cast<hipDoubleComplex*>(work), lwork, devInfo);
    #endif
}


/* ---------- gesvd ---------- */
cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverSgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverCgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *S,
                                  float *U,
                                  int ldu,
                                  float *VT,
                                  int ldvt,
                                  float *work,
                                  int lwork,
                                  float *rwork,
                                  int *info) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work and lwork as rocSOLVER does not need them
    return rocsolver_sgesvd(handle, convert_rocblas_svect(jobu), convert_rocblas_svect(jobvt),
                            m, n, A, lda, S, U, ldu, VT, ldvt, rwork, rocblas_outofplace,  // always out-of-place
                            info);
    #else
    return hipsolverSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info);
    #endif
}

cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *S,
                                  double *U,
                                  int ldu,
                                  double *VT,
                                  int ldvt,
                                  double *work,
                                  int lwork,
                                  double *rwork,
                                  int *info) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work and lwork as rocSOLVER does not need them
    return rocsolver_dgesvd(handle, convert_rocblas_svect(jobu), convert_rocblas_svect(jobvt),
                            m, n, A, lda, S, U, ldu, VT, ldvt, rwork, rocblas_outofplace,  // always out-of-place
                            info);
    #else
    return hipsolverDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info);
    #endif
}

cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  float *S,
                                  cuComplex *U,
                                  int ldu,
                                  cuComplex *VT,
                                  int ldvt,
                                  cuComplex *work,
                                  int lwork,
                                  float *rwork,
                                  int *info) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work and lwork as rocSOLVER does not need them
    return rocsolver_cgesvd(handle, convert_rocblas_svect(jobu), convert_rocblas_svect(jobvt),
                            m, n, reinterpret_cast<rocblas_float_complex*>(A), lda,
                            S, reinterpret_cast<rocblas_float_complex*>(U), ldu,
                            reinterpret_cast<rocblas_float_complex*>(VT), ldvt, rwork,
                            rocblas_outofplace,  // always out-of-place
                            info);
    #else
    return hipsolverCgesvd(handle, jobu, jobvt, m, n, reinterpret_cast<hipFloatComplex*>(A), lda,
                            S, reinterpret_cast<hipFloatComplex*>(U), ldu, reinterpret_cast<hipFloatComplex*>(VT),
                            ldvt, reinterpret_cast<hipFloatComplex*>(work), lwork, rwork, info);
    #endif
}

cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  double *S,
                                  cuDoubleComplex *U,
                                  int ldu,
                                  cuDoubleComplex *VT,
                                  int ldvt,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  double *rwork,
                                  int *info) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work and lwork as rocSOLVER does not need them
    return rocsolver_zgesvd(handle, convert_rocblas_svect(jobu), convert_rocblas_svect(jobvt),
                            m, n, reinterpret_cast<rocblas_double_complex*>(A), lda,
                            S, reinterpret_cast<rocblas_double_complex*>(U), ldu,
                            reinterpret_cast<rocblas_double_complex*>(VT), ldvt, rwork,
                            rocblas_outofplace,  // always out-of-place
                            info);
    #else
    return hipsolverZgesvd(handle, jobu, jobvt, m, n, reinterpret_cast<hipDoubleComplex*>(A), lda,
                            S, reinterpret_cast<hipDoubleComplex*>(U), ldu, reinterpret_cast<hipDoubleComplex*>(VT),
                            ldvt, reinterpret_cast<hipDoubleComplex*>(work), lwork, rwork, info);
    #endif
}


/* ---------- batched gesvd ---------- */
// Because rocSOLVER provides no counterpart for gesvdjBatched, we wrap its batched version directly.
typedef enum {
    CUSOLVER_EIG_MODE_NOVECTOR=0,
    CUSOLVER_EIG_MODE_VECTOR=1
} cusolverEigMode_t;

#if HIP_VERSION < 504
typedef void* gesvdjInfo_t;
#else
//typedef hipsolverEigMode_t cusolverEigMode_t;
typedef hipsolverGesvdjInfo_t gesvdjInfo_t;

static hipsolverEigMode_t convert_hipsolver_eigmode(cusolverEigMode_t mode) {
    return static_cast<hipsolverEigMode_t>(static_cast<int>(mode) + 201);
}

#endif

cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) {
  #if HIP_VERSION >= 504
    //return hipsolverDnCreateGesvdjInfo(info);
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    // should always success as rocSOLVER does not need it
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) {
  #if HIP_VERSION >= 504
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverDestroyGesvdjInfo(info);
  #else
    // should always success as rocSOLVER does not need it
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        const float *A,
        int lda,
        const float *S,
        const float *U,
        int ldu,
        const float *V,
        int ldv,
        int *lwork,
        gesvdjInfo_t params,
        int batchSize) {
  #if HIP_VERSION >= 504
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverSgesvdj_bufferSize(handle, convert_hipsolver_eigmode(jobz), m, n, A, lda, S, U, ldu, V, ldv, lwork,
    //                                    params, batchSize);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the bidiagonal matrix B associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * (m<n?m:n);  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        const double *A,
        int lda,
        const double *S,
        const double *U,
        int ldu,
        const double *V,
        int ldv,
        int *lwork,
        gesvdjInfo_t params,
        int batchSize) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverDgesvdj_bufferSize(handle, convert_hipsolver_eigmode(jobz), m, n, A, lda, S, U, ldu, V, ldv, lwork,
    //                                    params, batchSize);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the bidiagonal matrix B associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * (m<n?m:n);  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCgesvdjBatched_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        const cuComplex *A,
        int lda,
        const float *S,
        const cuComplex *U,
        int ldu,
        const cuComplex *V,
        int ldv,
        int *lwork,
        gesvdjInfo_t params,
        int batchSize) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverCgesvdj_bufferSize(handle, convert_hipsolver_eigmode(jobz), m, n,
    //                                    reinterpret_cast<const hipFloatComplex*>(A), lda, S,
    //                                    reinterpret_cast<const hipFloatComplex*>(U), ldu,
    //                                    reinterpret_cast<const hipFloatComplex*>(V), ldv, lwork,
    //                                    params, batchSize);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the bidiagonal matrix B associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * (m<n?m:n);  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZgesvdjBatched_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        const cuDoubleComplex *A,
        int lda,
        const double *S,
        const cuDoubleComplex *U,
        int ldu,
        const cuDoubleComplex *V,
        int ldv,
        int *lwork,
        gesvdjInfo_t params,
        int batchSize) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverZgesvdj_bufferSize(handle, convert_hipsolver_eigmode(jobz), m, n,
    //                                    reinterpret_cast<const hipDoubleComplex*>(A), lda, S,
    //                                    reinterpret_cast<const hipDoubleComplex*>(U), ldu,
    //                                    reinterpret_cast<const hipDoubleComplex*>(V), ldv, lwork,
    //                                    params, batchSize);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the bidiagonal matrix B associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * (m<n?m:n);  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSgesvdjBatched(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        float *A,
        int lda,
        float *S,
        float *U,
        int ldu,
        float *V,
        int ldv,
        float *work,
        int lwork,
        int *info,
        gesvdjInfo_t params,
        int batchSize) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    rocblas_svect leftv, rightv;
    rocblas_stride stU, stV;
    if (jobz == CUSOLVER_EIG_MODE_NOVECTOR) {
        leftv = rocblas_svect_none;
        rightv = rocblas_svect_none;
        stU = ldu * (m<n?m:n);
        stV = ldv * n;
    } else {  // CUSOLVER_EIG_MODE_VECTOR
        leftv = rocblas_svect_all;
        rightv = rocblas_svect_all;
        stU = ldu * m;
        stV = ldv * n;
    }
    return rocsolver_sgesvd_batched(handle, leftv, rightv,
                                    m, n, reinterpret_cast<float* const*>(A), lda,
                                    S, m<n?m:n,
                                    U, ldu, stU,
                                    V, ldv, stV,
                                    // since we can't pass in another array through the API, and work is unused,
                                    // we use it to store the temporary E array, to be discarded after calculation
                                    work, (m<n?m:n)-1,
                                    rocblas_outofplace, // always out-of-place
                                    info, batchSize);
    #else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverSgesvdjBatched(handle, convert_hipsolver_eigmode(jobz), m, n, A, lda,
    //                                S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
    #endif
}

cusolverStatus_t cusolverDnDgesvdjBatched(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        double *A,
        int lda,
        double *S,
        double *U,
        int ldu,
        double *V,
        int ldv,
        double *work,
        int lwork,
        int *info,
        gesvdjInfo_t params,
        int batchSize) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    rocblas_svect leftv, rightv;
    rocblas_stride stU, stV;
    if (jobz == CUSOLVER_EIG_MODE_NOVECTOR) {
        leftv = rocblas_svect_none;
        rightv = rocblas_svect_none;
        stU = ldu * (m<n?m:n);
        stV = ldv * n;
    } else {  // CUSOLVER_EIG_MODE_VECTOR
        leftv = rocblas_svect_all;
        rightv = rocblas_svect_all;
        stU = ldu * m;
        stV = ldv * n;
    }
    return rocsolver_dgesvd_batched(handle, leftv, rightv,
                                    m, n, reinterpret_cast<double* const*>(A), lda,
                                    S, m<n?m:n,
                                    U, ldu, stU,
                                    V, ldv, stV,
                                    // since we can't pass in another array through the API, and work is unused,
                                    // we use it to store the temporary E array, to be discarded after calculation
                                    work, (m<n?m:n)-1,
                                    rocblas_outofplace, // always out-of-place
                                    info, batchSize);
    #else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverDgesvdjBatched(handle, convert_hipsolver_eigmode(jobz), m, n, A, lda,
    //                                S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
    #endif
}

cusolverStatus_t cusolverDnCgesvdjBatched(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        cuComplex *A,
        int lda,
        float *S,
        cuComplex *U,
        int ldu,
        cuComplex *V,
        int ldv,
        cuComplex *work,
        int lwork,
        int *info,
        gesvdjInfo_t params,
        int batchSize) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    rocblas_svect leftv, rightv;
    rocblas_stride stU, stV;
    if (jobz == CUSOLVER_EIG_MODE_NOVECTOR) {
        leftv = rocblas_svect_none;
        rightv = rocblas_svect_none;
        stU = ldu * (m<n?m:n);
        stV = ldv * n;
    } else {  // CUSOLVER_EIG_MODE_VECTOR
        leftv = rocblas_svect_all;
        rightv = rocblas_svect_all;
        stU = ldu * m;
        stV = ldv * n;
    }
    return rocsolver_cgesvd_batched(handle, leftv, rightv,
                                    m, n, reinterpret_cast<rocblas_float_complex* const*>(A), lda,
                                    S, m<n?m:n,
                                    reinterpret_cast<rocblas_float_complex*>(U), ldu, stU,
                                    reinterpret_cast<rocblas_float_complex*>(V), ldv, stV,
                                    // since we can't pass in another array through the API, and work is unused,
                                    // we use it to store the temporary E array, to be discarded after calculation
                                    reinterpret_cast<float*>(work), (m<n?m:n)-1,
                                    rocblas_outofplace, // always out-of-place
                                    info, batchSize);
    #else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverCgesvdjBatched(handle, convert_hipsolver_eigmode(jobz), m, n, reinterpret_cast<hipFloatComplex*>(A), lda,
    //                                S, reinterpret_cast<hipFloatComplex*>(U), ldu, reinterpret_cast<hipFloatComplex*>(V), ldv,
    //                                reinterpret_cast<hipFloatComplex*>(work), lwork, info, params, batchSize);
    #endif
}

cusolverStatus_t cusolverDnZgesvdjBatched(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        int m,
        int n,
        cuDoubleComplex *A,
        int lda,
        double *S,
        cuDoubleComplex *U,
        int ldu,
        cuDoubleComplex *V,
        int ldv,
        cuDoubleComplex *work,
        int lwork,
        int *info,
        gesvdjInfo_t params,
        int batchSize) {
    #if HIP_VERSION < 309
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    rocblas_svect leftv, rightv;
    rocblas_stride stU, stV;
    if (jobz == CUSOLVER_EIG_MODE_NOVECTOR) {
        leftv = rocblas_svect_none;
        rightv = rocblas_svect_none;
        stU = ldu * (m<n?m:n);
        stV = ldv * n;
    } else {  // CUSOLVER_EIG_MODE_VECTOR
        leftv = rocblas_svect_all;
        rightv = rocblas_svect_all;
        stU = ldu * m;
        stV = ldv * n;
    }
    return rocsolver_zgesvd_batched(handle, leftv, rightv,
                                    m, n, reinterpret_cast<rocblas_double_complex* const*>(A), lda,
                                    S, m<n?m:n,
                                    reinterpret_cast<rocblas_double_complex*>(U), ldu, stU,
                                    reinterpret_cast<rocblas_double_complex*>(V), ldv, stV,
                                    // since we can't pass in another array through the API, and work is unused,
                                    // we use it to store the temporary E array, to be discarded after calculation
                                    reinterpret_cast<double*>(work), (m<n?m:n)-1,
                                    rocblas_outofplace, // always out-of-place
                                    info, batchSize);
    #else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
    //return hipsolverZgesvdjBatched(handle, convert_hipsolver_eigmode(jobz), m, n, reinterpret_cast<hipDoubleComplex*>(A), lda,
    //                                S, reinterpret_cast<hipDoubleComplex*>(U), ldu, reinterpret_cast<hipDoubleComplex*>(V), ldv,
    //                                reinterpret_cast<hipDoubleComplex*>(work), lwork, info, params, batchSize);
    #endif
}


/* ---------- gebrd ---------- */
cusolverStatus_t cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverSgebrd_bufferSize(handle, m, n, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDgebrd_bufferSize(handle, m, n, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverCgebrd_bufferSize(handle, m, n, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZgebrd_bufferSize(handle, m, n, Lwork);
  #else
    // this needs to return 0 because rocSolver does not rely on it
    *Lwork = 0;
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSgebrd(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *D,
                                  float *E,
                                  float *TAUQ,
                                  float *TAUP,
                                  float *Work,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work, lwork and devinfo as rocSOLVER does not need them
    return rocsolver_sgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP);
    #else
    return hipsolverSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
    #endif
}

cusolverStatus_t cusolverDnDgebrd(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *D,
                                  double *E,
                                  double *TAUQ,
                                  double *TAUP,
                                  double *Work,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work, lwork and devinfo as rocSOLVER does not need them
    return rocsolver_dgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP);
    #else
    return hipsolverDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
    #endif
}

cusolverStatus_t cusolverDnCgebrd(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  float *D,
                                  float *E,
                                  cuComplex *TAUQ,
                                  cuComplex *TAUP,
                                  cuComplex *Work,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work, lwork and devinfo as rocSOLVER does not need them
    return rocsolver_cgebrd(handle, m, n, reinterpret_cast<rocblas_float_complex*>(A),
                            lda, D, E, reinterpret_cast<rocblas_float_complex*>(TAUQ),
                            reinterpret_cast<rocblas_float_complex*>(TAUP));
    #else
    return hipsolverCgebrd(handle, m, n, reinterpret_cast<hipFloatComplex*>(A),
                            lda, D, E, reinterpret_cast<hipFloatComplex*>(TAUQ),
                            reinterpret_cast<hipFloatComplex*>(TAUP), reinterpret_cast<hipFloatComplex*>(Work), Lwork, devInfo);
    #endif
}

cusolverStatus_t cusolverDnZgebrd(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  double *D,
                                  double *E,
                                  cuDoubleComplex *TAUQ,
                                  cuDoubleComplex *TAUP,
                                  cuDoubleComplex *Work,
                                  int Lwork,
                                  int *devInfo) {
    #if HIP_VERSION < 306
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    // ignore work, lwork and devinfo as rocSOLVER does not need them
    return rocsolver_zgebrd(handle, m, n, reinterpret_cast<rocblas_double_complex*>(A),
                            lda, D, E, reinterpret_cast<rocblas_double_complex*>(TAUQ),
                            reinterpret_cast<rocblas_double_complex*>(TAUP));
    #else
    return hipsolverZgebrd(handle, m, n, reinterpret_cast<hipDoubleComplex*>(A),
                            lda, D, E, reinterpret_cast<hipDoubleComplex*>(TAUQ),
                            reinterpret_cast<hipDoubleComplex*>(TAUP), reinterpret_cast<hipDoubleComplex*>(Work), Lwork, devInfo);
    #endif
}


/* ---------- syevj ---------- */
#if HIP_VERSION < 540
typedef void* syevjInfo_t;
#else
typedef hipsolverSyevjInfo_t syevjInfo_t;
#endif

#if HIP_VERSION >= 402
static rocblas_evect convert_rocblas_evect(cusolverEigMode_t mode) {
    switch(mode) {
        // as of ROCm 4.2.0 rocblas_evect_tridiagonal is not supported
        case 0 /* CUSOLVER_EIG_MODE_NOVECTOR */: return rocblas_evect_none;
        case 1 /* CUSOLVER_EIG_MODE_VECTOR */  : return rocblas_evect_original;
        default: throw std::runtime_error("unrecognized mode");
    }
}
#endif

cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t *info) {
  #if HIP_VERSION >= 540
    return hipsolverCreateSyevjInfo(info);
  #else
    // TODO(leofang): set info to NULL? We don't use it anyway...
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info) {
  #if HIP_VERSION >= 540
    return hipsolverDestroySyevjInfo(info);
  #else
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo,
                                             int n,
                                             const float *A,
                                             int lda,
                                             const float *W,
                                             int *lwork,
                                             syevjInfo_t params) {
  #if HIP_VERSION >= 540
    return hipsolverSsyevj_bufferSize(handle, convert_hipsolver_eigmode(jobz),
                                        convert_to_hipsolverFill(uplo), n, const_cast<float*>(A), lda,
                                        const_cast<float*>(W), lwork, params);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the tridiagonal matrix T associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = n;  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo,
                                             int n,
                                             const double *A,
                                             int lda,
                                             const double *W,
                                             int *lwork,
                                             syevjInfo_t params) {
  #if HIP_VERSION >= 540
    return hipsolverDsyevj_bufferSize(handle, convert_hipsolver_eigmode(jobz),
                                        convert_to_hipsolverFill(uplo), n, const_cast<double*>(A), lda,
                                        const_cast<double*>(W), lwork, params);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the tridiagonal matrix T associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = n;  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCheevj_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo,
                                             int n,
                                             const cuComplex *A,
                                             int lda,
                                             const float *W,
                                             int *lwork,
                                             syevjInfo_t params) {
  #if HIP_VERSION >= 540
    return hipsolverCheevj_bufferSize(handle, convert_hipsolver_eigmode(jobz),
                                        convert_to_hipsolverFill(uplo), n, reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(A)),
                                        lda, const_cast<float*>(W), lwork, params);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the tridiagonal matrix T associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = n;  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZheevj_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo,
                                             int n,
                                             const cuDoubleComplex *A,
                                             int lda,
                                             const double *W,
                                             int *lwork,
                                             syevjInfo_t params) {
  #if HIP_VERSION >= 540
    return hipsolverZheevj_bufferSize(handle, convert_hipsolver_eigmode(jobz),
                                        convert_to_hipsolverFill(uplo), n, reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(A)),
                                        lda, const_cast<double*>(W), lwork, params);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the tridiagonal matrix T associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = n;  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSsyevj(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *W,
                                  float *work,
                                  int lwork,
                                  int *info,
                                  syevjInfo_t params) {
    #if HIP_VERSION < 402
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_ssyev(handle, convert_rocblas_evect(jobz), convert_rocblas_fill(uplo),
                           n, A, lda, W,
                           // since we can't pass in another array through the API, and work is unused,
                           // we use it to store the temporary E array, to be discarded after calculation
                           work,
                           info);
    #else
    return hipsolverSsyevj(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                            n, A, lda, W, work, lwork, info, params);
    #endif
}

cusolverStatus_t cusolverDnDsyevj(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *W,
                                  double *work,
                                  int lwork,
                                  int *info,
                                  syevjInfo_t params) {
    #if HIP_VERSION < 402
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_dsyev(handle, convert_rocblas_evect(jobz), convert_rocblas_fill(uplo),
                           n, A, lda, W,
                           // since we can't pass in another array through the API, and work is unused,
                           // we use it to store the temporary E array, to be discarded after calculation
                           work,
                           info);
    #else
    return hipsolverDsyevj(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                            n, A, lda, W, work, lwork, info, params);
    #endif
}

cusolverStatus_t cusolverDnCheevj(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  float *W,
                                  cuComplex *work,
                                  int lwork,
                                  int *info,
                                  syevjInfo_t params) {
    #if HIP_VERSION < 402
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_cheev(handle, convert_rocblas_evect(jobz), convert_rocblas_fill(uplo),
                           n, reinterpret_cast<rocblas_float_complex*>(A), lda, W,
                           // since we can't pass in another array through the API, and work is unused,
                           // we use it to store the temporary E array, to be discarded after calculation
                           reinterpret_cast<float*>(work),
                           info);
    #else
    return hipsolverCheevj(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                            n, reinterpret_cast<hipFloatComplex*>(A), lda, W, reinterpret_cast<hipFloatComplex*>(work),
                            lwork, info, params);
    #endif
}

cusolverStatus_t cusolverDnZheevj(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  double *W,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  int *info,
                                  syevjInfo_t params) {
    #if HIP_VERSION < 402
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_zheev(handle, convert_rocblas_evect(jobz), convert_rocblas_fill(uplo),
                           n, reinterpret_cast<rocblas_double_complex*>(A), lda, W,
                           // since we can't pass in another array through the API, and work is unused,
                           // we use it to store the temporary E array, to be discarded after calculation
                           reinterpret_cast<double*>(work),
                           info);
    #else
    return hipsolverZheevj(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                            n, reinterpret_cast<hipDoubleComplex*>(A), lda, W, reinterpret_cast<hipDoubleComplex*>(work),
                            lwork, info, params);
    #endif
}

/* ---------- batched syevj ---------- */
cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float *A,
                                                    int lda,
                                                    const float *W,
                                                    int *lwork,
                                                    syevjInfo_t params,
                                                    int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverSsyevjBatched_bufferSize(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                                                n, const_cast<float*>(A), lda, const_cast<float*>(W), lwork, params, batchSize);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the tridiagonal matrix T associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * n;  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double *A,
                                                    int lda,
                                                    const double *W,
                                                    int *lwork,
                                                    syevjInfo_t params,
                                                    int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverDsyevjBatched_bufferSize(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                                                n, const_cast<double*>(A), lda, const_cast<double*>(W), lwork, params, batchSize);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the tridiagonal matrix T associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * n;  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnCheevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const cuComplex *A,
                                                    int lda,
                                                    const float *W,
                                                    int *lwork,
                                                    syevjInfo_t params,
                                                    int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverCheevjBatched_bufferSize(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                                                n, reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(A)), lda,
                                                const_cast<float*>(W), lwork, params, batchSize);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the tridiagonal matrix T associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * n;  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnZheevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const double *W,
                                                    int *lwork,
                                                    syevjInfo_t params,
                                                    int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverZheevjBatched_bufferSize(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                                                n, reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(A)), lda,
                                                const_cast<double*>(W), lwork, params, batchSize);
  #else
    // rocSOLVER does not need extra workspace, but it needs to allocate memory for storing
    // the tridiagonal matrix T associated with A, which we don't need, so we use this workspace
    // to store it
    *lwork = batchSize * n;  // note: counts, not bytes!
    return rocblas_status_success;
  #endif
}

cusolverStatus_t cusolverDnSsyevjBatched(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo,
                                         int n,
                                         float *A,
                                         int lda,
                                         float *W,
                                         float *work,
                                         int lwork,
                                         int *info,
                                         syevjInfo_t params,
                                         int batchSize) {
    #if HIP_VERSION < 402
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_ssyev_batched(handle, convert_rocblas_evect(jobz), convert_rocblas_fill(uplo),
                                   n, reinterpret_cast<float* const*>(A), lda, W, n,
                                   // since we can't pass in another array through the API, and work is unused,
                                   // we use it to store the temporary E array, to be discarded after calculation
                                   work, n,
                                   info, batchSize);
    #else
    return hipsolverSsyevjBatched(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                                    n, A, lda, W, work, lwork, info, params, batchSize);
    #endif
}

cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo,
                                         int n,
                                         double *A,
                                         int lda,
                                         double *W,
                                         double *work,
                                         int lwork,
                                         int *info,
                                         syevjInfo_t params,
                                         int batchSize) {
    #if HIP_VERSION < 402
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_dsyev_batched(handle, convert_rocblas_evect(jobz), convert_rocblas_fill(uplo),
                                   n, reinterpret_cast<double* const*>(A), lda, W, n,
                                   // since we can't pass in another array through the API, and work is unused,
                                   // we use it to store the temporary E array, to be discarded after calculation
                                   work, n,
                                   info, batchSize);
    #else
    return hipsolverDsyevjBatched(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                                    n, A, lda, W, work, lwork, info, params, batchSize);
    #endif
}

cusolverStatus_t cusolverDnCheevjBatched(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo,
                                         int n,
                                         cuComplex *A,
                                         int lda,
                                         float *W,
                                         cuComplex *work,
                                         int lwork,
                                         int *info,
                                         syevjInfo_t params,
                                         int batchSize) {
    #if HIP_VERSION < 402
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_cheev_batched(handle, convert_rocblas_evect(jobz), convert_rocblas_fill(uplo),
                                   n, reinterpret_cast<rocblas_float_complex* const*>(A), lda, W, n,
                                   // since we can't pass in another array through the API, and work is unused,
                                   // we use it to store the temporary E array, to be discarded after calculation
                                   reinterpret_cast<float*>(work), n,
                                   info, batchSize);
    #else
    return hipsolverCheevjBatched(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                                    n, reinterpret_cast<hipFloatComplex*>(A), lda, W, reinterpret_cast<hipFloatComplex*>(work),
                                    lwork, info, params, batchSize);
    #endif
}

cusolverStatus_t cusolverDnZheevjBatched(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo,
                                         int n,
                                         cuDoubleComplex *A,
                                         int lda,
                                         double *W,
                                         cuDoubleComplex *work,
                                         int lwork,
                                         int *info,
                                         syevjInfo_t params,
                                         int batchSize) {
    #if HIP_VERSION < 402
    return rocblas_status_not_implemented;
    #elif HIP_VERSION < 540
    return rocsolver_zheev_batched(handle, convert_rocblas_evect(jobz), convert_rocblas_fill(uplo),
                                   n, reinterpret_cast<rocblas_double_complex* const*>(A), lda, W, n,
                                   // since we can't pass in another array through the API, and work is unused,
                                   // we use it to store the temporary E array, to be discarded after calculation
                                   reinterpret_cast<double*>(work), n,
                                   info, batchSize);
    #else
    return hipsolverZheevjBatched(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo),
                                    n, reinterpret_cast<hipDoubleComplex*>(A), lda, W, reinterpret_cast<hipDoubleComplex*>(work),
                                    lwork, info, params, batchSize);
    #endif
}


/* all of the stubs below are unsupported functions; the supported ones are moved to above */

typedef enum{} cusolverEigType_t;
#if HIP_VERSION >= 540
static hipsolverEigType_t convert_hipsolver_eigtype(cusolverEigType_t type) {
    return static_cast<hipsolverEigType_t>(static_cast<int>(type) + 211);
}
#endif
typedef void* cusolverSpHandle_t;
typedef void* cusparseMatDescr_t;

cusolverStatus_t cusolverSpGetStream(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpSetStream(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}


/* ---------- potrs ---------- */
cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle,
           cublasFillMode_t uplo,
           int n,
           int nrhs,
           const float *A,
           int lda,
           float *B,
           int ldb,
           int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverSpotrs(handle, convert_to_hipsolverFill(uplo), n, nrhs,
                            const_cast<float*>(A), lda, B, ldb, nullptr, 0, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle,
           cublasFillMode_t uplo,
           int n,
           int nrhs,
           const double *A,
           int lda,
           double *B,
           int ldb,
           int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverDpotrs(handle, convert_to_hipsolverFill(uplo), n, nrhs,
                            const_cast<double*>(A), lda, B, ldb, nullptr, 0, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCpotrs(cusolverDnHandle_t handle,
           cublasFillMode_t uplo,
           int n,
           int nrhs,
           const cuComplex *A,
           int lda,
           cuComplex *B,
           int ldb,
           int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverCpotrs(handle, convert_to_hipsolverFill(uplo), n, nrhs,
                            reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(A)), lda,
                            reinterpret_cast<hipFloatComplex*>(B), ldb, nullptr, 0, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t handle,
           cublasFillMode_t uplo,
           int n,
           int nrhs,
           const cuDoubleComplex *A,
           int lda,
           cuDoubleComplex *B,
           int ldb,
           int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverZpotrs(handle, convert_to_hipsolverFill(uplo), n, nrhs,
                            reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(A)), lda,
                            reinterpret_cast<hipDoubleComplex*>(B), ldb, nullptr, 0, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnSpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    float *Aarray[],
    int lda,
    float *Barray[],
    int ldb,
    int *info,
    int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverSpotrsBatched(handle, convert_to_hipsolverFill(uplo), n, nrhs, Aarray, lda,
                                    Barray, ldb, nullptr, 0, info, batchSize);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDpotrsBatched(cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    double *Aarray[],
    int lda,
    double *Barray[],
    int ldb,
    int *info,
    int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverDpotrsBatched(handle, convert_to_hipsolverFill(uplo), n, nrhs, Aarray, lda,
                                    Barray, ldb, nullptr, 0, info, batchSize);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCpotrsBatched(cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    cuComplex *Aarray[],
    int lda,
    cuComplex *Barray[],
    int ldb,
    int *info,
    int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverCpotrsBatched(handle, convert_to_hipsolverFill(uplo), n, nrhs, reinterpret_cast<hipFloatComplex**>(Aarray), lda,
                                    reinterpret_cast<hipFloatComplex**>(Barray), ldb, nullptr, 0, info, batchSize);
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZpotrsBatched(cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    cuDoubleComplex *Aarray[],
    int lda,
    cuDoubleComplex *Barray[],
    int ldb,
    int *info,
    int batchSize) {
  #if HIP_VERSION >= 540
    return hipsolverZpotrsBatched(handle, convert_to_hipsolverFill(uplo), n, nrhs, reinterpret_cast<hipDoubleComplex**>(Aarray), lda,
                                    reinterpret_cast<hipDoubleComplex**>(Barray), ldb, nullptr, 0, info, batchSize);
  #else
    return rocblas_status_not_implemented;
  #endif
}


/* ---------- sytrf ---------- */
cusolverStatus_t cusolverDnSsytrf_bufferSize(cusolverDnHandle_t handle,
                      int n,
                      float *A,
                      int lda,
                      int *Lwork ) {
  #if HIP_VERSION >= 540
    return hipsolverSsytrf_bufferSize(handle, n, A, lda, Lwork);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDsytrf_bufferSize(cusolverDnHandle_t handle,
                      int n,
                      double *A,
                      int lda,
                      int *Lwork ) {
  #if HIP_VERSION >= 540
    return hipsolverDsytrf_bufferSize(handle, n, A, lda, Lwork);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCsytrf_bufferSize(cusolverDnHandle_t handle,
                      int n,
                      cuComplex *A,
                      int lda,
                      int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverCsytrf_bufferSize(handle, n, reinterpret_cast<hipFloatComplex*>(A), lda, Lwork);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZsytrf_bufferSize(cusolverDnHandle_t handle,
                      int n,
                      cuDoubleComplex *A,
                      int lda,
                      int *Lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZsytrf_bufferSize(handle, n, reinterpret_cast<hipDoubleComplex*>(A), lda, Lwork);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnSsytrf(cusolverDnHandle_t handle,
           cublasFillMode_t uplo,
           int n,
           float *A,
           int lda,
           int *ipiv,
           float *work,
           int lwork,
           int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverSsytrf(handle, convert_to_hipsolverFill(uplo), n, A, lda, ipiv, work, lwork, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDsytrf(cusolverDnHandle_t handle,
           cublasFillMode_t uplo,
           int n,
           double *A,
           int lda,
           int *ipiv,
           double *work,
           int lwork,
           int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverDsytrf(handle, convert_to_hipsolverFill(uplo), n, A, lda, ipiv, work, lwork, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCsytrf(cusolverDnHandle_t handle,
           cublasFillMode_t uplo,
           int n,
           cuComplex *A,
           int lda,
           int *ipiv,
           cuComplex *work,
           int lwork,
           int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverCsytrf(handle, convert_to_hipsolverFill(uplo), n, reinterpret_cast<hipFloatComplex*>(A), lda,
                            ipiv, reinterpret_cast<hipFloatComplex*>(work), lwork, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZsytrf(cusolverDnHandle_t handle,
           cublasFillMode_t uplo,
           int n,
           cuDoubleComplex *A,
           int lda,
           int *ipiv,
           cuDoubleComplex *work,
           int lwork,
           int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverZsytrf(handle, convert_to_hipsolverFill(uplo), n, reinterpret_cast<hipDoubleComplex*>(A), lda,
                            ipiv, reinterpret_cast<hipDoubleComplex*>(work), lwork, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXgesvdjSetTolerance(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXgesvdjSetSortEig(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXgesvdjGetResidual(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXgesvdjGetSweeps(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnSgesvdj_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDgesvdj_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCgesvdj_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZgesvdj_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnSgesvdj(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDgesvdj(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCgesvdj(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZgesvdj(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}


cusolverStatus_t cusolverDnSgesvdaStridedBatched_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnSgesvdaStridedBatched(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZZgels_bufferSize(cusolverDnHandle_t handle,
    int m,
    int n,
    int nrhs,
    cuDoubleComplex* dA,
    int ldda,
    cuDoubleComplex* dB,
    int lddb,
    cuDoubleComplex* dX,
    int lddx,
    void* dwork,
    size_t* lwork_bytes) {
  #if HIP_VERSION >= 540
    return hipsolverZZgels_bufferSize(handle, m, n, nrhs, reinterpret_cast<hipDoubleComplex*>(dA), ldda,
            reinterpret_cast<hipDoubleComplex*>(dB), lddb, reinterpret_cast<hipDoubleComplex*>(dX), lddx,
            lwork_bytes);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZCgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZYgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZKgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCCgels_bufferSize(cusolverDnHandle_t handle,
    int m,
    int n,
    int nrhs,
    cuComplex* dA,
    int ldda,
    cuComplex* dB,
    int lddb,
    cuComplex * dX,
    int lddx,
    void* dwork,
    size_t* lwork_bytes) {
  #if HIP_VERSION >= 540
    return hipsolverCCgels_bufferSize(handle, m, n, nrhs, reinterpret_cast<hipFloatComplex*>(dA), ldda,
                                        reinterpret_cast<hipFloatComplex*>(dB), lddb, reinterpret_cast<hipFloatComplex*>(dX), lddx, lwork_bytes);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCYgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCKgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDDgels_bufferSize(cusolverDnHandle_t handle,
    int m,
    int n,
    int nrhs,
    double* dA,
    int ldda,
    double* dB,
    int lddb,
    double* dX,
    int lddx,
    void* dwork,
    size_t* lwork_bytes) {
  #if HIP_VERSION >= 540
    return hipsolverDDgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, lwork_bytes);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDSgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDXgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDHgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSSgels_bufferSize(cusolverDnHandle_t handle,
    int m,
    int n,
    int nrhs,
    float* dA,
    int ldda,
    float* dB,
    int lddb,
    float* dX,
    int lddx,
    void* dwork,
    size_t* lwork_bytes) {
  #if HIP_VERSION >= 540
    return hipsolverSSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, lwork_bytes);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSXgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSHgels_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZZgels(cusolverDnHandle_t handle,
        int m,
        int n,
        int nrhs,
        cuDoubleComplex* dA,
        int ldda,
        cuDoubleComplex* dB,
        int lddb,
        cuDoubleComplex* dX,
        int lddx,
        void* dWorkspace,
        size_t lwork_bytes,
        int* niter,
        int* dinfo) {
  #if HIP_VERSION >= 540
    return hipsolverZZgels(handle, m, n, nrhs, reinterpret_cast<hipDoubleComplex*>(dA), ldda,
                            reinterpret_cast<hipDoubleComplex*>(dB), lddb, reinterpret_cast<hipDoubleComplex*>(dX), lddx,
                            dWorkspace, lwork_bytes, niter, dinfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZCgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZYgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZKgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCCgels(cusolverDnHandle_t handle,
        int m,
        int n,
        int nrhs,
        cuComplex* dA,
        int ldda,
        cuComplex* dB,
        int lddb,
        cuComplex* dX,
        int lddx,
        void* dWorkspace,
        size_t lwork_bytes,
        int* niter,
        int* dinfo) {
  #if HIP_VERSION >= 540
    return hipsolverCCgels(handle, m, n, nrhs, reinterpret_cast<hipFloatComplex*>(dA), ldda,
                            reinterpret_cast<hipFloatComplex*>(dB), lddb, reinterpret_cast<hipFloatComplex*>(dX), lddx,
                            dWorkspace, lwork_bytes, niter, dinfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCYgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCKgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDDgels(cusolverDnHandle_t handle,
        int m,
        int n,
        int nrhs,
        double* dA,
        int ldda,
        double* dB,
        int lddb,
        double* dX,
        int lddx,
        void* dWorkspace,
        size_t lwork_bytes,
        int* niter,
        int* dinfo) {
  #if HIP_VERSION >= 540
    return hipsolverDDgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niter, dinfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDSgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDXgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDHgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSSgels(cusolverDnHandle_t handle,
        int m,
        int n,
        int nrhs,
        float* dA,
        int ldda,
        float* dB,
        int lddb,
        float* dX,
        int lddx,
        void * dWorkspace,
        size_t lwork_bytes,
        int * niter,
        int * dinfo) {
  #if HIP_VERSION >= 540
    return hipsolverSSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niter, dinfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSXgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSHgels(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverSsyevd_bufferSize(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo), n,
                                        const_cast<float*>(A), lda, const_cast<float*>(W), lwork);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverDsyevd_bufferSize(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo), n,
                                        const_cast<double*>(A), lda, const_cast<double*>(W), lwork);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *W,
    int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverCheevd_bufferSize(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo), n,
                                        reinterpret_cast<hipFloatComplex*>(const_cast<cuComplex*>(A)), lda,
                                        const_cast<float*>(W), lwork);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork) {
  #if HIP_VERSION >= 540
    return hipsolverZheevd_bufferSize(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo), n,
                                        reinterpret_cast<hipDoubleComplex*>(const_cast<cuDoubleComplex*>(A)), lda,
                                        const_cast<double*>(W), lwork);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *W,
    float *work,
    int lwork,
    int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverSsyevd(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo), n, A, lda,
                            W, work, lwork, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W,
    double *work,
    int lwork,
    int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverDsyevd(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo), n, A, lda,
                            W, work, lwork, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnCheevd(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *W,
    cuComplex *work,
    int lwork,
    int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverCheevd(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo), n,
                            reinterpret_cast<hipFloatComplex*>(A), lda, W,
                            reinterpret_cast<hipFloatComplex*>(work), lwork, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo) {
  #if HIP_VERSION >= 540
    return hipsolverZheevd(handle, convert_hipsolver_eigmode(jobz), convert_to_hipsolverFill(uplo), n,
                            reinterpret_cast<hipDoubleComplex*>(A), lda, W,
                            reinterpret_cast<hipDoubleComplex*>(work), lwork, devInfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info,
    double tolerance) {
  #if HIP_VERSION >= 540
    return hipsolverXsyevjSetTolerance(info, tolerance);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info,
    int max_sweeps) {
  #if HIP_VERSION >= 540
    return hipsolverXsyevjSetMaxSweeps(info, max_sweeps);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info,
    int sort_eig) {
  #if HIP_VERSION >= 540
    return hipsolverXsyevjSetSortEig(info, sort_eig);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle,
    syevjInfo_t info,
    double *residual) {
  #if HIP_VERSION >= 540
    return hipsolverXsyevjGetResidual(handle, info, residual);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle,
    syevjInfo_t info,
    int *executed_sweeps) {
  #if HIP_VERSION >= 540
    return hipsolverXsyevjGetSweeps(handle, info, executed_sweeps);
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnZZgesv_bufferSize(cusolverDnHandle_t handle,
    int n,
    int nrhs,
    cuDoubleComplex* dA,
    int ldda,
    int* dipiv,
    cuDoubleComplex* dB,
    int lddb,
    cuDoubleComplex* dX,
    int lddx,
    void * dwork,
    size_t* lwork_bytes) {
  #if HIP_VERSION >= 540
    return hipsolverZZgesv_bufferSize(handle, n, nrhs, reinterpret_cast<hipDoubleComplex*>(dA), ldda, dipiv,
                                        reinterpret_cast<hipDoubleComplex*>(dB), lddb, reinterpret_cast<hipDoubleComplex*>(dX), lddx,
                                        lwork_bytes);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZCgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZYgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZKgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCCgesv_bufferSize(cusolverDnHandle_t handle,
    int n,
    int nrhs,
    cuComplex* dA,
    int ldda,
    int * dipiv,
    cuComplex* dB,
    int lddb,
    cuComplex* dX,
    int lddx,
    void* dwork,
    size_t* lwork_bytes) {
  #if HIP_VERSION >= 540
    return hipsolverCCgesv_bufferSize(handle, n, nrhs, reinterpret_cast<hipFloatComplex*>(dA), ldda,
                                        dipiv, reinterpret_cast<hipFloatComplex*>(dB), lddb,
                                        reinterpret_cast<hipFloatComplex*>(dX), lddx, lwork_bytes);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCYgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCKgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDDgesv_bufferSize(cusolverDnHandle_t handle,
    int n,
    int nrhs,
    double* dA,
    int     ldda,
    int   * dipiv,
    double* dB,
    int     lddb,
    double* dX,
    int     lddx,
    void  * dwork,
    size_t* lwork_bytes) {
  #if HIP_VERSION >= 540
    return hipsolverDDgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, lwork_bytes);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDSgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDXgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDHgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSSgesv_bufferSize(cusolverDnHandle_t handle,
    int n,
    int nrhs,
    float* dA,
    int    ldda,
    int  * dipiv,
    float* dB,
    int    lddb,
    float* dX,
    int    lddx,
    void * dwork,
    size_t * lwork_bytes) {
  #if HIP_VERSION >= 540
    return hipsolverSSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, lwork_bytes);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSXgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSHgesv_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZZgesv(cusolverDnHandle_t handle,
        int n,
        int nrhs,
        cuDoubleComplex*   dA,
        int                ldda,
        int            *   dipiv,
        cuDoubleComplex*   dB,
        int                lddb,
        cuDoubleComplex*   dX,
        int                lddx,
        void           *   dWorkspace,
        size_t             lwork_bytes,
        int            *   niter,
        int            *   dinfo) {
  #if HIP_VERSION >= 540
    return hipsolverZZgesv(handle, n, nrhs, reinterpret_cast<hipDoubleComplex*>(dA), ldda, dipiv,
                            reinterpret_cast<hipDoubleComplex*>(dB), lddb, reinterpret_cast<hipDoubleComplex*>(dX), lddx,
                            dWorkspace, lwork_bytes, niter, dinfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZCgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZYgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnZKgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCCgesv(cusolverDnHandle_t handle,
        int n,
        int nrhs,
        cuComplex*   dA,
        int          ldda,
        int      *   dipiv,
        cuComplex*   dB,
        int          lddb,
        cuComplex*   dX,
        int          lddx,
        void     *   dWorkspace,
        size_t       lwork_bytes,
        int      *   niter,
        int      *   dinfo) {
  #if HIP_VERSION >= 540
    return hipsolverCCgesv(handle, n, nrhs, reinterpret_cast<hipFloatComplex*>(dA), ldda, dipiv,
                            reinterpret_cast<hipFloatComplex*>(dB), lddb, reinterpret_cast<hipFloatComplex*>(dX), lddx,
                            dWorkspace, lwork_bytes, niter, dinfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCYgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnCKgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDDgesv(cusolverDnHandle_t handle,
        int  n,
        int  nrhs,
        double*   dA,
        int       ldda,
        int   *   dipiv,
        double*   dB,
        int       lddb,
        double*   dX,
        int       lddx,
        void  *   dWorkspace,
        size_t    lwork_bytes,
        int   *   niter,
        int   *   dinfo) {
  #if HIP_VERSION >= 540
    return hipsolverDDgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niter, dinfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDSgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDXgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnDHgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSSgesv(cusolverDnHandle_t handle,
        int n,
        int nrhs,
        float * dA,
        int     ldda,
        int   * dipiv,
        float * dB,
        int     lddb,
        float * dX,
        int     lddx,
        void  * dWorkspace,
        size_t  lwork_bytes,
        int   * niter,
        int   * dinfo) {
  #if HIP_VERSION >= 540
    return hipsolverSSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niter, dinfo);
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSXgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}
cusolverStatus_t cusolverDnSHgesv(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXsyevd_bufferSize(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverDnXsyevd(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpCreate(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpDestroy(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpScsrlsvqr(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpDcsrlsvqr(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpCcsrlsvqr(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpZcsrlsvqr(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpScsrlsvchol(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpDcsrlsvchol(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpCcsrlsvchol(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpZcsrlsvchol(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpScsreigvsi(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpDcsreigvsi(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpCcsreigvsi(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

cusolverStatus_t cusolverSpZcsreigvsi(...) {
  #if HIP_VERSION >= 540
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
  #else
    return rocblas_status_not_implemented;
  #endif
}

} // extern "C"

#endif // #ifdef INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H
