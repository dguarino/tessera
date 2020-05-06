# tessera
A small set of helper functions to ease the developing of spiking neural networks in [PyNN](http://neuralensemble.org/docs/PyNN/index.html) by having all parameters, command line interpretation (also for parameter searches), and analysis, in one place. 

*tessera* is inspired from, and you can take it as a *very* simplified version of, [mozaik](https://github.com/antolikjan/mozaik). Hence its name, which is italian for the *tile* of a mosaic. 

## no need to install
*tessera* is meant to be a drop-in solution for cluster and neuromorphic deployment where you can only upload files but not install new libraries/packages/...

*tessera* is assuming that PyNN is installed with at least NEST. To ease the pain of installing a full stack with NEURON, NEST, numpy, matplotlib ... a docker image can do the job, for example neuralensembles/simulationx (see docker.com and the `Dockerfile` in *tessera*).

Therefore, **you don't** ~~`python setup.py install tessera`~~. You just clone it, enter the folder, and start driving PyNN.

## files
*tessera* is made of two files:

* helpers.py - contains all the routines to drive PyNN to define models and stimuli, run simulations (also for parameter searches), collect state values and save them as results ready to be analysed.
* run.py - contains the code to interpret various shell commands 

You then run your analysis as you want, based on PyNN's output files in the neo format. 

My personal set of function is in:

* analysis.py - additional routines for the analysis of simulation results

## how to
*tessera* is a dictionary of parameters used to drive PyNN. 

I built it to help me focus only on the core elements required to develop neural networks: the model parameters. 

In the directory `examples` you will find a series of files that illustrate the method I use to develop spiking neural networks: 
1. find electrophysiology data for the **cells**
  * fit PyNN single cells using the same experimental protocols
2. find electrophysiology data for the **synapses**
  * fit PyNN synapses using the same experimental protocols
3. find electrophysiology data for the **network**
  * build a PyNN network using: 
    * cells and synapses developed above
    * the experimental protocol closest to the available data

### how to: thalamocortical network
Say you want to model slow-wave or asynchronous irregular firing regimes in a thalamo-cortical network of spiking neurons.

Start by reproducing single cell responses to injection of excitatory and inhibitory current pulses, and to one-to-one synaptic interaction. Examples of these tasks are in the parameter files for [thalamic relay cell](examples/TC_response.py), [thalamic reticular cell](examples/RE_response.py), [cortical regular spiking cell](examples/RS_response.py), [cortical fast spiking cell](examples/FS_response.py). 

Then, use the parameters found by fitting in-vitro studies above to build networks, using additional in-vitro and in-vivo intra- and extra-cellular measurements to constrain the building procedure. Examples of increasing complexity are in the parameter files for "in-vitro slice"-like networks ([thalamic network](examples/thalamic_delta_spindles.py), and [cortical network](examples/cortical_SW.py)), and "in-vivo"-like networks ([thalamocortical network (slow waves)](examples/thalamocortical_SW_delta.py), [thalamocortical network (spindles)](examples/thalamocortical_SW_Spindles.py)).

#### how to: current injection in cortical pyramidal cell
Say you want to match the single cell response to current injections of regular spiking pyramidal neurons in different input resistance states. 

The parameter file [cortical regular spiking cell](examples/RS_response.py) contains all it is required to reproduce the behaviour of this type of cell, as found in the paper of McCormick and Prince "Mechanisms of action of acetylcholine in the guinea-pig cerebral cortex in vitro" (1986).

As cell excitability model, I chose the Adaptive exponential Integrate and Fire neuron (AdEx for short, see PyNN [docs](http://neuralensemble.org/docs/PyNN/reference/neuronmodels.html#pyNN.standardmodels.cells.EIF_cond_alpha_isfa_ista) for it). In the parameter file, the section `Populations` (line 25) collects all the PyNN populations of neurons that will be used. In this example there are only two populations (we will see the population `ext` later). The population `cell` is made of only one AdEx neuron (line 41):

    'type': sim.EIF_cond_alpha_isfa_ista,

Some parameters, like membrane capacitance and spike treshold, will be set from sources like [neuroelectro](https://www.neuroelectro.org/neuron/111/) (you will find the reference for each parameter in a comment on the same line of the parameter file).

Some other parameters, like time constant of leak conductance and resting potential, are found by iterating over the protocols in the paper and systematically searching within a range of values (we will see how below). 

For example, the Figure 3 illustrates the RS cell response to excitatory and inhibitory current pulses. In the parameter file, in the section `Injections` (line 145), you will find an essential reproduction of the experimental protocol of figure 3B.

    'source' : sim.StepCurrentSource,
    'amplitude' : [.15, .0], # depolarising
    # 'amplitude' : [-.15, .0], # hyperpolarising
    'start' : [200., 320.], # 

The parameter `source` is a proxyfor the PyNN type of injection source, with its corresponding `amplitude` and times of injection (`start`).

##### how to: run this example

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

#### how to: PSP in cortical pyramidal cell
Say you want to match the single cell EPSP or IPSP response of regular spiking pyramidal neurons against different input resistance states. 

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

Move to the folder "tessera" checked out from github and build the image

```
docker build -t tessera .
```

Check the existence of the image

```
docker images
```

Start a container with the "tessera" image
```
docker run -i -t tessera /bin/bash
```

And to allow for development bind-mount your local files in the container

```
docker run -v `pwd`:`pwd` -w `pwd` -i -t tessera /bin/bash

```

Then check the container id (or name)

```
docker ps
```

