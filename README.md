# goSPL Docker container
## Docker container with pre- & post-processing libraries

The container is built with `Anaconda` and `python v3.10`.

On top of the libraries required for **goSPL**, the following main ones are available:

- xarray, rioxarray, xarray-spatial
- pygmt, cartopy, pygplates
- xesmf, metpy
- stripy, meshio, meshplex, triangle
- vtk, hdf5, netCDF4

### Note

If you need additional libraries you could install them from the Jupyter terminal directly (when opening the terminal change the shell to `bash` by using running `bash`) within the container by using either the `conda install` command or `pip install` command.

## Install goSPL Docker image

Docker is an open platform for developing, shipping, and running applications. You can install it from:

+ [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

Once you have installed Docker Desktop for your operating system then enable the docker comand line (`Docker CLI`) and pull the goSPL Docker image from the terminal using the following:

```bash
docker pull geodels/gosplcontainer:latest
```

Once pulled, you can run the **gosplcontainer** using:
```bash
docker run -it -p 8888:8888 -d -v localDIR:/notebooks geodels/gosplcontainer:latest
```
where `localDIR` is the directory that contains the Jupyter Notebooks for your simulation.

Once Docker is running, you could open the Jupyter notebooks on a web browser at the following address:
+ [http://localhost:8888/](http://localhost:8888/)

This should open in your `localDIR` directory.

### Additional useful Docker commands

List the running containers:
```bash
docker ps
```

Use `docker stop xxxx` where `xxxxx` is the CONTAINER ID from the list obtained with the previous command. 

To stop all running containers:
```bash
docker stop $(docker ps -a -q)
```

To remove the **gosplcontainer**:
```bash
docker image rm  geodels/gosplcontainer:latest
```

