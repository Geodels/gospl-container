FROM continuumio/anaconda3
MAINTAINER "Tristan Salles"
RUN apt-get update 
RUN /opt/conda/bin/conda config --env --add channels defaults 
RUN /opt/conda/bin/conda config --env --add channels conda-forge 
RUN /opt/conda/bin/conda config --env --add channels anaconda
RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda install python=3.10 && \
    /opt/conda/bin/conda install anaconda-client && \
    /opt/conda/bin/conda install jupyter -y && \
    /opt/conda/bin/conda install numpy pandas scikit-learn scikit-image matplotlib seaborn h5py -y 
RUN /opt/conda/bin/conda install -c conda-forge pygplates
RUN /opt/conda/bin/conda install compilers petsc4py llvm-openmp netCDF4 -y && \
    /opt/conda/bin/conda install mpi4py matplotlib numpy-indexed pysheds -y
RUN pip install  richdem descartes pyevtk vtk stripy triangle
RUN pip install meshio==4.4.6
RUN pip install triangle
RUN /opt/conda/bin/conda install packaging -y
RUN /opt/conda/bin/conda install gmt -y
RUN /opt/conda/bin/conda install pygmt -y
RUN /opt/conda/bin/conda install numba -y
RUN /opt/conda/bin/conda install -c conda-forge meshplex -y
RUN apt-get install -y m4
RUN /opt/conda/bin/conda install rioxarray -y
RUN /opt/conda/bin/conda install boost-cpp -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN /opt/conda/bin/conda install make -y
RUN /opt/conda/bin/conda install -c conda-forge esmf -y
RUN /opt/conda/bin/conda install -c conda-forge xesmf -y
RUN /opt/conda/bin/conda install -c conda-forge ncurses -y
RUN /opt/conda/bin/conda install -c conda-forge metpy -y
RUN /opt/conda/bin/conda install -c conda-forge xarray-spatial -y
RUN pip install wget cmocean cartopy
RUN pip install numpy==1.23.4
RUN pip uninstall stripy -y
RUN pip install stripy
ENV ESMFMKFILE=/opt/conda/lib/esmf.mk
# See file packages/README-docker for the installation of pnetcdf and parallel-dbscan
COPY packages/pnetcdf/lib/libpnetcdf.* /opt/conda/lib/
COPY packages/pnetcdf/lib/pnetcdf.pc /opt/conda/lib/pkgconfig/
COPY packages/pnetcdf/include/* /opt/conda/include/
COPY packages/pDBSCAN/dbscan /opt/conda/bin/
RUN chmod +x /opt/conda/bin/dbscan
# Install goSPL
COPY gospl /root/gospl
RUN cd /root/gospl/; python3 setup.py install
# Define shared volume folder
RUN ["mkdir", "notebooks"]
# BASH command
ADD conf/bashrc-term /root/.bashrc
COPY conf/.jupyter /root/.jupyter
COPY run_jupyter.sh /
# Jupyter port
EXPOSE 8888
# Store notebooks in this mounted directory
VOLUME /notebooks
CMD ["/run_jupyter.sh"]