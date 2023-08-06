cmake_minimum_required(VERSION 3.26)
project(tf_custom_op CXX C CUDA)

set(CMAKE_CXX_STANDARD 14)

# prepare CUDA env
FIND_PACKAGE(CUDA REQUIRED)
MESSAGE(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
MESSAGE(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
MESSAGE(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
MESSAGE(STATUS "CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
SET(CUDA_CUBLASLT_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublasLt.so")
MESSAGE(STATUS "CUDA_CUBLASLT_LIBRARIES: ${CUDA_CUBLASLT_LIBRARIES}")
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
SET(CUDA_NVCC_FLAGS "-std=c++14" "-Xcompiler=-fno-gnu-unique")
SET(CMAKE_CUDA_STANDARD 14)
IF (DEFINED GPU_ARCHS)
    MESSAGE(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
    SEPARATE_ARGUMENTS(GPU_ARCHS)
ELSE ()
    LIST(APPEND GPU_ARCHS
            61
            70
            75
            )
    STRING(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
    IF (IS_ARM)
        # Xavier (SM72) only supported for aarch64.
        LIST(APPEND GPU_ARCHS 72)
    ENDIF ()

    IF (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
        # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
        LIST(APPEND GPU_ARCHS 80)
    ELSE ()
        message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
    ENDIF ()

    MESSAGE(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
ENDIF ()
# Generate SASS for each architecture
FOREACH (arch ${GPU_ARCHS})
    SET(GENCODES "${GENCODES}" "-gencode arch=compute_${arch},code=sm_${arch}")
ENDFOREACH ()
SET(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHS})
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations --expt-relaxed-constexpr")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${GENCODES}")
MESSAGE(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
MESSAGE(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")


# prepare tensorflow env.
# TensorFlow 网站上提供的二进制 pip 软件包使用 gcc4 构建，该编译器使用旧版 ABI。
# 添加 -D_GLIBCXX_USE_CXX11_ABI=0，使库与旧版 ABI 兼容。
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DGOOGLE_CUDA=1)
# tf.sysconfig.get_include()获取
set(TF_INCLUDE_DIR "/usr/local/lib/python3.7/dist-packages/tensorflow/include")
# tf.sysconfig.get_lib()获取
set(TF_LIB_DIR "/usr/local/lib/python3.7/dist-packages/tensorflow")
include_directories(${TF_INCLUDE_DIR})

add_library(tf_custom_op SHARED kernel_example.cc kernel_example.cu)
target_link_directories(tf_custom_op BEFORE PUBLIC ${TF_LIB_DIR})
target_link_libraries(tf_custom_op tensorflow_framework ${CUDA_LIBRARIES})
