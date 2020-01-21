# tessera
Thin layer of helper functions on top of PyNN to have all parameters, recorders, command line interpretation (also for parameter searches), and analysis in one place.

## structure
tessera is made of three files:

* helpers.py - contains all the routines to drive PyNN in creating a simulation
* run.py - contains the code to interpret the command line
* analysis.py - additional routines for the analysis of simulation results

tessera is assuming a full PyNN installation with at least the latest NEST. It usually drops perfectly in a docker container created from the neuralensembles/simulationx docker image (see below for example code).


## usage examples

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

