cmake_minimum_required(VERSION 3.18)
project(pyba LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)



find_package(pybind11 REQUIRED)
# Use the fetched Eigen and Ceres
set(Eigen3_DIR "$ENV{CONDA_PREFIX}/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

pybind11_add_module(pyceres_bind cpp/bind.cc)
target_link_libraries(pyceres_bind PRIVATE ceres Eigen3::Eigen)

# Install the module to the pyba package
install(TARGETS pyceres_bind LIBRARY DESTINATION pyba) 