all:
	#g++ -I/usr/include/eigen3 -O2	singularity_analyzer_gmsh_multifile_Q.cc -o singtrack_gmsh_multifile_Q -lgmsh
	#g++ -I/usr/include/eigen3 -O2	singularity_analyzer_gmsh_multifile_Q_Gouraud.cc -o singtrack_gmsh_multifile_Q_Gouraud -lgmsh
	g++  -I/usr/include/eigen3 -O3 -march=native -fopenmp -std=c++17 singularity_analyzer_Q_Gouraud_optimized.cc -o singtrack_multifile_Q_Gouraud_optimized -lgmsh
