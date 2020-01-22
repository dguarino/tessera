# tessera
Thin layer of helper functions on top of PyNN to have all parameters, recorders, command line interpretation (also for parameter searches), and analysis in one place.

## no need to install
Tessera is meant to be a set of helper scripts, to develop spiking neuron models straight out of a PyNN installation and as a drop-in for cluster deployment where you can only upload files but not install.

Therefore, you don't need to `python setup.py install tessera`. You just clone it, enter the folder, and start simulating.

However, if you really want, you can install it to derive your own classes using it as a module.

## structure
tessera is made of three files:

* helpers.py - contains all the routines to drive PyNN in creating a simulation
* run.py - contains the code to interpret the command line
* analysis.py - additional routines for the analysis of simulation results

tessera is assuming a full PyNN installation with at least the latest NEST. It usually drops perfectly in a docker container created from the neuralensembles/simulationx docker image (see below for example code).

## parameters file
no hassles with tessera, the only file you usually need to modify is the parameter file to drive PyNN.

You will find a series of example files in the directory `example`. Here below some explanations.



## usage examples
Enter the tessera folder

```
cd tessera
```

Run simple code

```
# python run.py --folder test --params epsp_response.py nest
```

Search example:

```
# python run.py --folder EPSPsearch --params epsp_response.py --search search.py --map yes nest
# python run.py --folder IPSPsearch --params ipsp_response.py --search search.py --map yes nest
# python plot_map.py
```

Analysis only example

```
# python run.py --folder EPSPsearch --params epsp_response.py --search search.py --analysis true nest
```

## Docker Image

Directly derived from: https://hub.docker.com/r/neuralensemble/simulationx/

With:

* shell environment with NEST 2.14, NEURON 7.5, and PyNN 0.9 installed.
* The Python 2.7 version provides Brian 1.4, the Python 3.4 version provides Brian 2.
* IPython, scipy, matplotlib and OpenMPI are also installed.

## Basic use

Start docker daemon

```
sudo systemctl restart docker
```

Enable the current user to launch docker images

```
sudo usermod -a -G docker $USER
```

Move to the folder "neuromod" checked out from github and build the image

```
docker build -t neuromod .
```

Check the existence of the image

```
docker images
```

Start a container with the "neuromod" image
```
docker run -i -t neuromod /bin/bash
```

And to allow for development bind-mount your local files in the container

```
docker run -v `pwd`:`pwd` -w `pwd` -i -t neuromod /bin/bash

```

Then check the container id (or name)

```
docker ps
```

