cpdef enum:
    # hipJitInputType
    hipJitInputTypeBin = 0
    hipJitInputTypePtx = 1
    hipJitInputTypeFatBinary = 2
    hipJitInputTypeObject = 3
    hipJitInputTypeLibrary = 4
    CU_JIT_INPUT_NVVM = 5

    # hipJitOption
    hipJitOptionMaxRegisters = 0
    hipJitOptionThreadsPerBlock = 1
    hipJitOptionWallTime = 2
    hipJitOptionInfoLogBuffer = 3
    hipJitOptionInfoLogBufferSizeBytes = 4
    hipJitOptionErrorLogBuffer = 5
    hipJitOptionErrorLogBufferSizeBytes = 6
    hipJitOptionOptimizationLevel = 7
    hipJitOptionTargetFromContext = 8
    hipJitOptionTarget = 9
    hipJitOptionFallbackStrategy = 10
    hipJitOptionGenerateDebugInfo = 11
    hipJitOptionLogVerbose = 12
    hipJitOptionGenerateLineInfo = 13
    hipJitOptionCacheMode = 14
    hipJitOptionSm3xOpt = 15
    hipJitOptionFastCompile = 16
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19
    CU_JIT_LTO = 20
    CU_JIT_FTZ = 21
    CU_JIT_PREC_DIV = 22
    CU_JIT_PREC_SQRT = 23
    CU_JIT_FMA = 24
    hipJitOptionNumOptions = 25

    # hipFuncAttribute_t
    hipFuncAttributeMaxThreadsPerBlocks = 0
    hipFuncAttributeSharedSizeBytes = 1
    hipFuncAttributeConstSizeBytes = 2
    hipFuncAttributeLocalSizeBytes = 3
    hipFuncAttributeNumRegs = 4
    hipFuncAttributePtxVersion = 5
    hipFuncAttributeBinaryVersion = 6
    hipFuncAttributeCacheModeCA = 7
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9

    # hipError_t
    hipErrorInvalidValue = 1

    # hipArray_format
    HIP_AD_FORMAT_UNSIGNED_INT8 = 0x01
    HIP_AD_FORMAT_UNSIGNED_INT16 = 0x02
    HIP_AD_FORMAT_UNSIGNED_INT32 = 0x03
    HIP_AD_FORMAT_SIGNED_INT8 = 0x08
    HIP_AD_FORMAT_SIGNED_INT16 = 0x09
    HIP_AD_FORMAT_SIGNED_INT32 = 0x0a
    HIP_AD_FORMAT_HALF = 0x10
    HIP_AD_FORMAT_FLOAT = 0x20

    # hipAddress_mode
    HIP_TR_ADDRESS_MODE_WRAP = 0
    HIP_TR_ADDRESS_MODE_CLAMP = 1
    HIP_TR_ADDRESS_MODE_MIRROR = 2
    HIP_TR_ADDRESS_MODE_BORDER = 3

    # hipTextureFilterMode
    hipFilterModePoint = 0
    hipFilterModeLinear = 1

    # Constants
    HIP_TRSA_OVERRIDE_FORMAT = 0x01

    HIP_TRSF_READ_AS_INTEGER = 0x01
    HIP_TRSF_NORMALIZED_COORDINATES = 0x02
    HIP_TRSF_SRGB = 0x10
