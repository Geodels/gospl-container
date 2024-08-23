# goSPL Docker container

## Docker container with pre- & post-processing libraries

The container is built with `Anaconda` and contains all the dependencies and configuration files required to run the [examples](https://github.com/Geodels/goSPL-examples).


##### Note

If you need additional libraries you could install them from the Jupyter terminal directly, then activate the goSPL environment (`conda activate gospl`) and then use either the `conda install` command or `pip install` command.

## goSPL Docker image

##### Pulling the image

Once you have installed Docker on your system, you can ``pull`` the
[goSPL official image](https://hub.docker.com/u/geodels) as follow::

```bash
  docker pull geodels/gospl:2024.09.01
```
##### Starting the container from a terminal

You can then start a docker container (an instance of
an image)::

```bash
  docker run -it -p 8888:8888 -d -v localDIR:/notebooks
```
where `localDIR` is the directory that contains the examples folder `goSPL-examples`.

Once Docker is running, you could open the Jupyter notebooks on a web browser at the following address: `http://localhost:8888 <http://localhost:8888>`_. Going into the `/notebooks` folder you will access your ``localDIR`` directory.

To run goSPL, you will need to use the terminal from the Jupyter interface. To activate the goSPL environment where all the libraries are installed you will have to run the following command:
```bash
  conda activate gospl
```

Depending on your operating system, you will be able to configure the docker application to set your resources: CPUs, memory, swap, or Disk image size. This will improve the performance of the run.

> Note that you could use the Dashboard from Docker instead of passing through the terminal to download the goSPL Docker image.