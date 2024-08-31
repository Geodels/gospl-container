# docker build --platform linux/amd64 -t geodels/gospl:2024.09.01 .
FROM continuumio/anaconda3:latest 
# @sha256:bd2af590d39a5d1b590cd6ad2abab37ae386b7e2a9b9d91e110d3d82074f3af9
LABEL org.opencontainers.image.authors="tristan.salles@sydney.edu.au"

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y libopenblas-dev liblapack-dev

# Create conda environment
RUN conda create -y -n gospl python=3.11.8 pip numpy
RUN echo "source activate gospl" > ~/.bashrc
ENV PATH=/opt/conda/envs/gospl/bin:$PATH
RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda config --env --add channels defaults \
    && conda config --env --add channels conda-forge \
    && conda config --env --add channels anaconda

# Compilers libraries
RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install compilers numpy==1.26.4 -y \
    && conda install compilers mpi4py -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install meson-python -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install dask -y \
    && apt-get install -y libgl1

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install scipy==1.13.1 pandas==2.2.2 h5py==3.11.0 -y \
    && conda install numba ruamel.yaml -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install gflex meshplex numpy-indexed jupyter -y

# Dependencies for post- & pre- processing
RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install pyevtk==1.6.0 -y \
    && conda install rasterio==1.3.9 -y 

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \  
    && conda install pyvista==0.44.0 -y \
    && conda install xarray==2024.6.0 -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install shapely==2.0.4 seaborn==0.13.2 -y \
    && pip install stripy

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install xesmf==0.8.5 -y \
    && conda install mpas_tools==0.33.0 -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install rioxarray==0.15.5 uxarray -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install pysheds==0.4 pyproj==3.6.1 -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install gmt pygmt==0.12.0 -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install pysqlite3 --y \
    && pip install perlin_noise

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install petsc4py -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install pysqlite3

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install sqlite -y

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && conda install cmake -y 

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && pip install pyinterp

# Install isoFlex
COPY packages/isoFlex /root/isoFlex
RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && cd /root/isoFlex/; pip install .

# Install goSPL
COPY packages/gospl /root/gospl
RUN conda init bash \
    && . ~/.bashrc \
    && conda activate gospl \
    && cd /root/gospl/; pip install .

# Define shared volume folder
RUN ["mkdir", "notebooks"]

# BASH command
COPY conf/.jupyter /root/.jupyter
COPY run_jupyter.sh /
ADD conf/bashrc-term /root/.bashrc

# Jupyter port
EXPOSE 8888
# Store notebooks in this mounted directory
VOLUME /notebooks
CMD ["/run_jupyter.sh"]