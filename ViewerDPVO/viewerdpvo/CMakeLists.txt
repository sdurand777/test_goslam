

# path to proper Pangolin
set(PANGOLIN_INCLUDE_DIR "/home/ivm/GO-SLAM/Pangolin/include")
set(PANGOLIN_LIB_DIR "/home/ivm/GO-SLAM/Pangolin/build")

# optional libraries
find_package(Pangolin REQUIRED)



if (Pangolin_FOUND)
    message(STATUS "Pangolin found at ${Pangolin_DIR}")
else()
    message(FATAL_ERROR "Pangolin not found at ${Pangolin_DIR}")
endif()


message(STATUS "CMakeLists de viewerdpvo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
#
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
#
# # Définir les options spécifiques pour nvcc
# set(CUDA_NVCC_FLAGS
#     ${CUDA_NVCC_FLAGS} 
#     -O3
#     -gencode=arch=compute_60,code=sm_60
#     -gencode=arch=compute_61,code=sm_61
#     -gencode=arch=compute_70,code=sm_70
#     -gencode=arch=compute_75,code=sm_75
#     -gencode=arch=compute_80,code=sm_80
#     -gencode=arch=compute_86,code=sm_86
# )
#
#
# set(CUDA_ARCH_BIN "75 80 86 89")

# Reset CUDA architectures and flags
unset(CMAKE_CUDA_ARCHITECTURES CACHE)
unset(CUDA_NVCC_FLAGS CACHE)

# Set desired CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")

# Define CUDA NVCC flags correctly
set(CUDA_NVCC_FLAGS "")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--generate-code=arch=compute_89,code=[compute_89,sm_89]")

string(REPLACE ";" " " CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

# Ensure CUDA flags are set
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS}")




#pybind11_add_module(dpviewerx viewer.cpp viewer_cuda.cu)
pybind11_add_module(viewerdpvo viewer.cpp viewer_cuda.cu)
include_directories(${Pangolin_INCLUDE_DIRS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(dpviewerx PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
# target_link_libraries(dpviewerx PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${Pangolin_LIBRARIES})
target_compile_definitions(viewerdpvo PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(viewerdpvo PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${Pangolin_LIBRARIES})

