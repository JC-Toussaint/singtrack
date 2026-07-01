cmake_minimum_required(VERSION 3.16)

project(fg-singtrack
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
# Executable : fg-singtrack
# ============================================================================

add_executable(fg-singtrack
    fg-singtrack.cc
)

target_link_libraries(fg-singtrack
    PRIVATE
        Eigen3::Eigen
        ${GMSH_LIBRARY}
)

# ============================================================================
# Executable : fg-singtrack_Gouraud
# ============================================================================

add_executable(fg-singtrack_Gouraud
    fg-singtrack_Gouraud.cc
)

target_link_libraries(fg-singtrack_Gouraud
    PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        ${GMSH_LIBRARY}
)
