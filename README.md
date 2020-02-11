# tessera
Thin layer of helper functions on top of PyNN to have all parameters, recorders, command line interpretation (also for parameter searches), and analysis in one place.

## no need to install
Tessera is meant to be a set of helper scripts, to develop spiking neuron models straight out of a PyNN installation and as a drop-in solution for cluster deployment where you can only upload files but not install libraries/packages/...

tessera is assuming that PyNN is installed with at least NEST. To ease the pain of installing a full stack from ubuntu to NEURON and NEST and numpy, and matplotlib ... a docker image can do the job, for example neuralensembles/simulationx (see docker.com and `Dockerfile`).

Therefore, **you don't need to** `python setup.py install tessera`. You just clone it, enter the folder, and start using the underlaying simulators through PyNN.

## files
tessera is made of two files:

* helpers.py - contains all the routines to drive PyNN to define models and stimuli, run simulations (also for parameter searches), collect state values and save them as results ready to be analysed.
* run.py - contains the code to interpret various shell commands 

You then run your analysis as you want, based on PyNN's output files in the neo format. 

My personal set of function is in:

* analysis.py - additional routines for the analysis of simulation results

## parameters
no hassles with tessera, you usually only need to modify a dictionary of parameters to drive PyNN.

You will find a series of examples in the directory `example`. Here below some explanations.

Say you want to model slow-wave sleep conditions in a network of spiking neurons.

My approach is that first, start by understanding single cell response properties to excitatory and inhibitory current pulses; second, use the parameters found by fitting in-vitro studies to constrain a network of similar units by also using additional in-vitro and in-vivo intra- and extra-cellular measurements.

Let's start with in-vitro single cell protocols.

### PSP in cortical pyramidal cell
Say you want to match the single cell EPSP or IPSP response of regular spiking pyramidal neurons against different input resistance states (as in McCormick and Pape 1986). 

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

