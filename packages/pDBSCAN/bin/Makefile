# --------------------------------------------------
# Makefile for Parallel DBSCAN implementation
#
# Set the Paths for:
# * PnetCDF library dir
# * MPI library dir (tested with openmpi and mpich)
# * BOOST library dir
# 
# --------------------------------------------------

# Files Folder
SRC = ../src

# Dependencies
# ----------------
PNET_DIR = /usr/local
MPI_DIR = /opt/conda
BOOST_DIR = /opt/conda

# Default Test parameters
# ----------------
i?=../datasets/clus50k.bin
m?=5
e?=25
output?=out_clustered.nc

# Compiler
# ----------------
CC = $(MPI_DIR)/bin/mpicc
EXE = dbscan

# Automatic Flags
# ----------------
OPTFLAGS = -O3
CFLAGS= -c -g $(OPTFLAGS) -I$(PNET_DIR)/include -I$(MPI_DIR)/include -I$(BOOST_DIR)/include -I$(SRC)
LDFLAGS=-g $(OPTFLAGS) -Wl,-rpath,$(PNET_DIR)/lib -L$(PNET_DIR)/lib
LIBS= -lstdc++ -lpnetcdf -lm
OBJS = $(patsubst $(SRC)/%.cpp,%.o,$(wildcard $(SRC)/*.cpp))

# Link rule
all: $(OBJS)
	$(CC) $(LDFLAGS) *.o -o $(EXE) $(LIBS)

# Generic module rule
%.o: $(SRC)/%.cpp
	$(CC) $(CFLAGS) $< -o $@

# File dependencies
clusters.o: $(SRC)/clusters.cpp
dbscan.o: $(SRC)/dbscan.cpp
geometric_partitioning.o: $(SRC)/geometric_partitioning.cpp
kdtree2.o: $(SRC)/kdtree2.cpp $(SRC)/kdtree2.hpp
utils.o: $(SRC)/utils.cpp
mpi_main.o: $(SRC)/mpi_main.cpp

# Clear dir
clean:
	rm -f *.o $(EXE)
