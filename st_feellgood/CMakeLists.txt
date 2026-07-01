cmake_minimum_required(VERSION 3.16)

project(singtrack
    VERSION 1.0
    LANGUAGES CXX
)

# ============================================================================
# C++
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimisation en Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================================
# Dependencies
# ============================================================================

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

# Recherche de la bibliothèque Gmsh
find_library(GMSH_LIBRARY gmsh REQUIRED)

# ============================================================================
# Executable : singtrack_Gouraud
# ============================================================================

add_executable(singtrack_Gouraud
    singtrack_Gouraud.cc
)

target_link_libraries(singtrack_Gouraud
    PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        ${GMSH_LIBRARY}
)

