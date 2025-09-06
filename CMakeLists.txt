cmake_minimum_required(VERSION 3.18)
project(pyba LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

include(FetchContent)

#find_package(pybind11 REQUIRED)
# Download pybind11 at configure time
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.5   # pick a stable release
)
FetchContent_MakeAvailable(pybind11)
# Use the fetched Eigen and Ceres
#set(Eigen3_DIR "$ENV{CONDA_PREFIX}/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)


set(GTSAM_USE_SYSTEM_EIGEN ON)
FetchContent_Declare(
  gtsam
  GIT_REPOSITORY  https://github.com/borglab/gtsam.git
  GIT_TAG        4.3a0   # specific version
)

# Actually download it
FetchContent_MakeAvailable(gtsam)

pybind11_add_module(pyceres_bind cpp/bind.cc)
target_link_libraries(pyceres_bind PRIVATE ceres Eigen3::Eigen gtsam)

# Install the module to the correct location
# This will be installed to the pyba package directory
install(TARGETS pyceres_bind LIBRARY DESTINATION pyba) 