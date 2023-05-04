cpdef enum:
    hipMemoryTypeHost = 1
    hipMemoryTypeDevice = 2

    hipIpcMemLazyEnablePeerAccess = 1

    hipMemAttachGlobal = 1
    hipMemAttachHost = 2
    hipMemAttachSingle = 4

    hipMemAdviseSetReadMostly = 1
    hipMemAdviseUnsetReadMostly = 2
    hipMemAdviseSetPreferredLocation = 3
    hipMemAdviseUnsetPreferredLocation = 4
    hipMemAdviseSetAccessedBy = 5
    hipMemAdviseUnsetAccessedBy = 6

    HIP_R_32F = 0  # 32 bit real
    HIP_R_64F = 1  # 64 bit real
    HIP_R_16F = 2  # 16 bit real
    HIP_R_8I = 3  # 8 bit real as a signed integer
    HIP_C_32F = 4  # 32 bit complex
    HIP_C_64F = 5  # 64 bit complex
    HIP_C_16F = 6  # 16 bit complex
    HIP_C_8I = 7  # 8 bit complex as a pair of signed integers
    HIP_R_8U = 8  # 8 bit real as a signed integer
    HIP_C_8U = 9  # 8 bit complex as a pair of signed integers

    # CUDA Limits
    hipLimitStackSize = 0x00
    hipLimitPrintfFifoSize = 0x01
    hipLimitMallocHeapSize = 0x02
    hipLimitDevRuntimeSyncDepth = 0x03
    hipLimitDevRuntimePendingLaunchCount = 0x04
    cudaLimitMaxL2FetchGranularity = 0x05

    # hipChannelFormatKind
    hipChannelFormatKindSigned = 0
    hipChannelFormatKindUnsigned = 1
    hipChannelFormatKindFloat = 2
    hipChannelFormatKindNone = 3

    # CUDA array flags
    hipArrayDefault = 0
    # hipArrayLayered = 1
    hipArraySurfaceLoadStore = 2
    # hipArrayCubemap = 4
    # hipArrayTextureGather = 8

    # hipResourceType
    hipResourceTypeArray = 0
    hipResourceTypeMipmappedArray = 1
    hipResourceTypeLinear = 2
    hipResourceTypePitch2D = 3

    # hipTextureAddressMode
    hipAddressModeWrap = 0
    hipAddressModeClamp = 1
    hipAddressModeMirror = 2
    hipAddressModeBorder = 3

    # hipTextureFilterMode
    hipFilterModePoint = 0
    hipFilterModeLinear = 1

    # hipTextureReadMode
    hipReadModeElementType = 0
    hipReadModeNormalizedFloat = 1

    # cudaMemPoolAttr
    # ----- added since 11.2 -----
    cudaMemPoolReuseFollowEventDependencies = 0x1
    cudaMemPoolReuseAllowOpportunistic = 0x2
    cudaMemPoolReuseAllowInternalDependencies = 0x3
    cudaMemPoolAttrReleaseThreshold = 0x4
    # ----- added since 11.3 -----
    cudaMemPoolAttrReservedMemCurrent = 0x5
    cudaMemPoolAttrReservedMemHigh = 0x6
    cudaMemPoolAttrUsedMemCurrent = 0x7
    cudaMemPoolAttrUsedMemHigh = 0x8

    # cudaMemAllocationType
    cudaMemAllocationTypePinned = 0x1

    # cudaMemAllocationHandleType
    cudaMemHandleTypeNone = 0x0
    cudaMemHandleTypePosixFileDescriptor = 0x1
    # cudaMemHandleTypeWin32 = 0x2
    # cudaMemHandleTypeWin32Kmt = 0x4

    # cudaMemLocationType
    cudaMemLocationTypeDevice = 1


# This was a legacy mistake: the prefix "cuda" should have been removed
# so that we can directly assign their C counterparts here. Now because
# of backward compatibility and no flexible Cython macro (IF/ELSE), we
# have to duplicate the enum. (CUDA and HIP use different values!)
IF CUPY_HIP_VERSION > 0:
    # separate in groups of 10 for easier counting...
    cpdef enum:
        hipDeviceAttributeMaxThreadsPerBlock = 0
        hipDeviceAttributeMaxBlockDimX
        hipDeviceAttributeMaxBlockDimY
        hipDeviceAttributeMaxBlockDimZ
        hipDeviceAttributeMaxGridDimX
        hipDeviceAttributeMaxGridDimY
        hipDeviceAttributeMaxGridDimZ
        hipDeviceAttributeMaxSharedMemoryPerBlock
        hipDeviceAttributeTotalConstantMemory
        hipDeviceAttributeWarpSize

        hipDeviceAttributeMaxRegistersPerBlock
        hipDeviceAttributeClockRate
        hipDeviceAttributeMemoryClockRate
        hipDeviceAttributeMemoryBusWidth
        hipDeviceAttributeMultiprocessorCount
        hipDeviceAttributeComputeMode
        hipDeviceAttributeL2CacheSize
        hipDeviceAttributeMaxThreadsPerMultiProcessor
        # The following are exposed as "deviceAttributeCo..."
        # hipDeviceAttributeComputeCapabilityMajor
        # hipDeviceAttributeComputeCapabilityMinor

        hipDeviceAttributeConcurrentKernels = 20
        hipDeviceAttributePciBusId
        hipDeviceAttributePciDeviceId
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
        hipDeviceAttributeIsMultiGpuBoard
        hipDeviceAttributeIntegrated
        cudaDevAttrCooperativeLaunch
        cudaDevAttrCooperativeMultiDeviceLaunch
        hipDeviceAttributeMaxTexture1DWidth
        hipDeviceAttributeMaxTexture2DWidth

        hipDeviceAttributeMaxTexture2DHeight
        hipDeviceAttributeMaxTexture3DWidth
        hipDeviceAttributeMaxTexture3DHeight
        hipDeviceAttributeMaxTexture3DDepth
        # The following attributes do not exist in CUDA and cause segfualts
        # if we try to access them
        # hipDeviceAttributeHdpMemFlushCntl
        # hipDeviceAttributeHdpRegFlushCntl
        hipDeviceAttributeMaxPitch = 36
        hipDeviceAttributeTextureAlignment
        hipDeviceAttributeTexturePitchAlignment
        hipDeviceAttributeKernelExecTimeout

        hipDeviceAttributeCanMapHostMemory
        hipDeviceAttributeEccEnabled
        cudaDevAttrMemoryPoolsSupported = 0
        # The following attributes do not exist in CUDA
        # hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
        # hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
        # hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
        # hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem

        # The rest do not have HIP correspondence...
        # TODO(leofang): should we expose them anyway, with a value -1 to
        # indicate they cannot be used in HIP?
        # hipDeviceAttributeGpuOverlap
        # hipDeviceAttributeMaxTexture2DLayeredWidth
        # hipDeviceAttributeMaxTexture2DLayeredHeight
        # hipDeviceAttributeMaxTexture2DLayeredLayers
        # hipDeviceAttributeSurfaceAlignment
        # hipDeviceAttributeTccDriver
        # hipDeviceAttributeAsyncEngineCount
        # hipDeviceAttributeUnifiedAddressing
        # hipDeviceAttributeMaxTexture1DLayeredWidth
        # hipDeviceAttributeMaxTexture1DLayeredLayers
        # hipDeviceAttributeMaxTexture2DGatherWidth
        # hipDeviceAttributeMaxTexture2DGatherHeight
        # hipDeviceAttributeMaxTexture3DWidthAlternate
        # hipDeviceAttributeMaxTexture3DHeightAlternate
        # hipDeviceAttributeMaxTexture3DDepthAlternate
        # hipDeviceAttributePciDomainId
        # hipDeviceAttributeMaxTextureCubemapWidth
        # hipDeviceAttributeMaxTextureCubemapLayeredWidth
        # hipDeviceAttributeMaxTextureCubemapLayeredLayers
        # hipDeviceAttributeMaxSurface1DWidth
        # hipDeviceAttributeMaxSurface2DWidth
        # hipDeviceAttributeMaxSurface2DHeight
        # hipDeviceAttributeMaxSurface3DWidth
        # hipDeviceAttributeMaxSurface3DHeight
        # hipDeviceAttributeMaxSurface3DDepth
        # hipDeviceAttributeMaxSurface1DLayeredWidth
        # hipDeviceAttributeMaxSurface1DLayeredLayers
        # hipDeviceAttributeMaxSurface2DLayeredWidth
        # hipDeviceAttributeMaxSurface2DLayeredHeight
        # hipDeviceAttributeMaxSurface2DLayeredLayers
        # hipDeviceAttributeMaxSurfaceCubemapWidth
        # hipDeviceAttributeMaxSurfaceCubemapLayeredWidth
        # hipDeviceAttributeMaxSurfaceCubemapLayeredLayers
        # hipDeviceAttributeMaxTexture1DLinearWidth
        # hipDeviceAttributeMaxTexture2DLinearWidth
        # hipDeviceAttributeMaxTexture2DLinearHeight
        # hipDeviceAttributeMaxTexture2DLinearPitch
        # hipDeviceAttributeMaxTexture2DMipmappedWidth
        # hipDeviceAttributeMaxTexture2DMipmappedHeight
        # hipDeviceAttributeMaxTexture1DMipmappedWidth
        # hipDeviceAttributeStreamPrioritiesSupported
        # hipDeviceAttributeGlobalL1CacheSupported
        # hipDeviceAttributeLocalL1CacheSupported
        # hipDeviceAttributeMaxRegistersPerMultiprocessor
        # hipDeviceAttributeMultiGpuBoardGroupID
        # hipDeviceAttributeHostNativeAtomicSupported
        # hipDeviceAttributeSingleToDoublePrecisionPerfRatio
        # hipDeviceAttributeComputePreemptionSupported
        # hipDeviceAttributeCanUseHostPointerForRegisteredMem
        # cudaDevAttrReserved92
        # cudaDevAttrReserved93
        # cudaDevAttrReserved94
        # cudaDevAttrMaxSharedMemoryPerBlockOptin
        # cudaDevAttrCanFlushRemoteWrites
        # cudaDevAttrHostRegisterSupported
    IF CUPY_HIP_VERSION >= 310:
        cpdef enum:
            # hipDeviceAttributeAsicRevision  # does not exist in CUDA
            hipDeviceAttributeManagedMemory = 47
            cudaDevAttrDirectManagedMemAccessFromHost
            hipDeviceAttributeConcurrentManagedAccess

            hipDeviceAttributePageableMemoryAccess
            cudaDevAttrPageableMemoryAccessUsesHostPageTables
ELSE:
    # For CUDA/RTD
    cpdef enum:
        hipDeviceAttributeMaxThreadsPerBlock = 1
        hipDeviceAttributeMaxBlockDimX
        hipDeviceAttributeMaxBlockDimY
        hipDeviceAttributeMaxBlockDimZ
        hipDeviceAttributeMaxGridDimX
        hipDeviceAttributeMaxGridDimY
        hipDeviceAttributeMaxGridDimZ
        hipDeviceAttributeMaxSharedMemoryPerBlock
        hipDeviceAttributeTotalConstantMemory
        hipDeviceAttributeWarpSize
        hipDeviceAttributeMaxPitch
        hipDeviceAttributeMaxRegistersPerBlock
        hipDeviceAttributeClockRate
        hipDeviceAttributeTextureAlignment
        hipDeviceAttributeGpuOverlap
        hipDeviceAttributeMultiprocessorCount
        hipDeviceAttributeKernelExecTimeout
        hipDeviceAttributeIntegrated
        hipDeviceAttributeCanMapHostMemory
        hipDeviceAttributeComputeMode
        hipDeviceAttributeMaxTexture1DWidth
        hipDeviceAttributeMaxTexture2DWidth
        hipDeviceAttributeMaxTexture2DHeight
        hipDeviceAttributeMaxTexture3DWidth
        hipDeviceAttributeMaxTexture3DHeight
        hipDeviceAttributeMaxTexture3DDepth
        hipDeviceAttributeMaxTexture2DLayeredWidth
        hipDeviceAttributeMaxTexture2DLayeredHeight
        hipDeviceAttributeMaxTexture2DLayeredLayers
        hipDeviceAttributeSurfaceAlignment
        hipDeviceAttributeConcurrentKernels
        hipDeviceAttributeEccEnabled
        hipDeviceAttributePciBusId
        hipDeviceAttributePciDeviceId
        hipDeviceAttributeTccDriver
        hipDeviceAttributeMemoryClockRate
        hipDeviceAttributeMemoryBusWidth
        hipDeviceAttributeL2CacheSize
        hipDeviceAttributeMaxThreadsPerMultiProcessor
        hipDeviceAttributeAsyncEngineCount
        hipDeviceAttributeUnifiedAddressing
        hipDeviceAttributeMaxTexture1DLayeredWidth
        hipDeviceAttributeMaxTexture1DLayeredLayers  # = 43; 44 is missing
        hipDeviceAttributeMaxTexture2DGatherWidth = 45
        hipDeviceAttributeMaxTexture2DGatherHeight
        hipDeviceAttributeMaxTexture3DWidthAlternate
        hipDeviceAttributeMaxTexture3DHeightAlternate
        hipDeviceAttributeMaxTexture3DDepthAlternate
        hipDeviceAttributePciDomainId
        hipDeviceAttributeTexturePitchAlignment
        hipDeviceAttributeMaxTextureCubemapWidth
        hipDeviceAttributeMaxTextureCubemapLayeredWidth
        hipDeviceAttributeMaxTextureCubemapLayeredLayers
        hipDeviceAttributeMaxSurface1DWidth
        hipDeviceAttributeMaxSurface2DWidth
        hipDeviceAttributeMaxSurface2DHeight
        hipDeviceAttributeMaxSurface3DWidth
        hipDeviceAttributeMaxSurface3DHeight
        hipDeviceAttributeMaxSurface3DDepth
        hipDeviceAttributeMaxSurface1DLayeredWidth
        hipDeviceAttributeMaxSurface1DLayeredLayers
        hipDeviceAttributeMaxSurface2DLayeredWidth
        hipDeviceAttributeMaxSurface2DLayeredHeight
        hipDeviceAttributeMaxSurface2DLayeredLayers
        hipDeviceAttributeMaxSurfaceCubemapWidth
        hipDeviceAttributeMaxSurfaceCubemapLayeredWidth
        hipDeviceAttributeMaxSurfaceCubemapLayeredLayers
        hipDeviceAttributeMaxTexture1DLinearWidth
        hipDeviceAttributeMaxTexture2DLinearWidth
        hipDeviceAttributeMaxTexture2DLinearHeight
        hipDeviceAttributeMaxTexture2DLinearPitch
        hipDeviceAttributeMaxTexture2DMipmappedWidth
        hipDeviceAttributeMaxTexture2DMipmappedHeight
        # The following are exposed as "deviceAttributeCo..."
        # hipDeviceAttributeComputeCapabilityMajor  # = 75
        # hipDeviceAttributeComputeCapabilityMinor  # = 76
        hipDeviceAttributeMaxTexture1DMipmappedWidth = 77
        hipDeviceAttributeStreamPrioritiesSupported
        hipDeviceAttributeGlobalL1CacheSupported
        hipDeviceAttributeLocalL1CacheSupported
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
        hipDeviceAttributeMaxRegistersPerMultiprocessor
        hipDeviceAttributeManagedMemory
        hipDeviceAttributeIsMultiGpuBoard
        hipDeviceAttributeMultiGpuBoardGroupID
        hipDeviceAttributeHostNativeAtomicSupported
        hipDeviceAttributeSingleToDoublePrecisionPerfRatio
        hipDeviceAttributePageableMemoryAccess
        hipDeviceAttributeConcurrentManagedAccess
        hipDeviceAttributeComputePreemptionSupported
        hipDeviceAttributeCanUseHostPointerForRegisteredMem
        cudaDevAttrReserved92
        cudaDevAttrReserved93
        cudaDevAttrReserved94
        cudaDevAttrCooperativeLaunch
        cudaDevAttrCooperativeMultiDeviceLaunch
        cudaDevAttrMaxSharedMemoryPerBlockOptin
        cudaDevAttrCanFlushRemoteWrites
        cudaDevAttrHostRegisterSupported
        cudaDevAttrPageableMemoryAccessUsesHostPageTables
        cudaDevAttrDirectManagedMemAccessFromHost  # = 101
        # added since CUDA 11.0
        cudaDevAttrMaxBlocksPerMultiprocessor = 106
        cudaDevAttrReservedSharedMemoryPerBlock = 111
        # added since CUDA 11.1
        cudaDevAttrSparseCudaArraySupported = 112
        cudaDevAttrHostRegisterReadOnlySupported = 113
        # added since CUDA 11.2
        cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114
        cudaDevAttrMemoryPoolsSupported = 115
        # added since CUDA 11.3
        cudaDevAttrGPUDirectRDMASupported
        cudaDevAttrGPUDirectRDMAFlushWritesOptions
        cudaDevAttrGPUDirectRDMAWritesOrdering
        cudaDevAttrMemoryPoolSupportedHandleTypes
