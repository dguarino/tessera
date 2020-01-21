# ------------------------------------------------------------------------------
# Usage:
# Examples

# start the docker image:
# $ docker run -v `pwd`:`pwd` -w `pwd` -i -t thalamus /bin/bash

# Run simple code
# python run.py --folder test --params epsp_response.py nest

# Search Example:
# python run.py --folder EPSPsearch --params epsp_response.py --search search.py --map yes nest
# python run.py --folder IPSPsearch --params ipsp_response.py --search search.py --map yes nest
# python plot_map.py

# Analysis Example
# python run.py --folder EPSPsearch --params epsp_response.py --search search.py --analysis true nest

# ./execute.bash
# ------------------------------------------------------------------------------
{
    'run_time': 10000, # ms
    'dt': 0.1, # ms

    'Populations' : {
        'ext' : {
            'n' : 1,
            'type': sim.SpikeSourcePoisson,
            'cellparams' : {
                'start':0.0,
                'rate':50.,
                'duration':100.0
            }
        },

        # THALAMUS
        'tc' : {
            'n': 100,
            'type' :  sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(dx=1.0, dy=1.0, fill_order='random', rng=sim.NumpyRNG(seed=13886)),
            'cellparams' : {
                'tau_syn_E'  : 5.0,   # ms
                'tau_syn_I'  : 5.0,   # ms
                'tau_refrac' : 2.5,   # ms, refractory period (ReinagelReid2000)
                'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold (Destexhe2009)
                # 'v_spike'    : 20.0,  # mV, spike detection
                'v_thresh'   : -50.0, # mV, fixed spike threshold (https://www.neuroelectro.org/neuron/190/)
                'cm'         : 0.16,  # nF, tot membrane capacitance (Bloomfield Hamos Sherman 1987)
                # Spindles
                # 'tau_m'      : 17.,  # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                # 'v_rest'     : -63.0, # mV, resting potential
                # 'v_reset'    : -48.0, # mV, reset after spike
                # 'a'          : 26.,    # nS, spike-frequency adaptation
                # 'b'          : .02,   # nA, increment to the adaptation variable
                # 'tau_w'      : 270.0, # ms, time constant of adaptation variable
                # Delta
                'tau_m'      : 18.,   # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                'v_rest'     : -65.0, # mV, resting potential (McCormick Pape 1988)
                # 'v_rest'     : -67.0, # mV, resting potential (McCormick Pape 1988)
                'v_reset'    : -46.0, # mV, reset after spike
                'a'          : 28.,   # nS, spike-frequency adaptation
                'b'          : .02,   # nA, increment to the adaptation variable
                'tau_w'      : 270.0, # ms, time constant of adaptation variable
            }
        },
        're' : {
            'n': 100,
            'type' :  sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(dx=1.0, dy=1.0, fill_order='random', rng=sim.NumpyRNG(seed=13886)),
            'cellparams' : {
                'tau_syn_E'  : 5.0,   # ms
                'tau_syn_I'  : 5.0,   # ms
                'tau_refrac' : 2.5,   # ms, refractory period (ReinagelReid2000)
                'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold (Destexhe2009)
                # 'v_spike'   : 20.0,   # mV, spike detection
                'v_thresh'   : -50.0, # mV, fixed spike threshold (https://www.neuroelectro.org/neuron/190/)
                'cm'         : 0.20,  # nF, tot membrane capacitance (https://www.neuroelectro.org/neuron/190/, Uhlrich Cucchiaro Humphrey Sherman 1991)
                # Spindles
                # 'tau_m'      : 15.,  # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                # 'v_rest'     : -70.0, # mV, resting potential
                # 'v_reset'    : -41.0, # mV, reset after spike (McCormick1992, Toubul and Brette 2008)
                # 'a'          : 28.,   # nS, 
                # 'b'          : .02,     # nA, spike-dependent adaptation
                # 'tau_w'      : 270.0, # ms, time constant of adaptation variable
                # Delta  -  McCormickPape1990
                'tau_m'      : 15.,  # ms, time constant of leak conductance (cm/gl)
                'v_rest'     : -65.0, # mV, resting potential (McCormick Prince 1986)
                'v_reset'    : -41.0, # mV, reset after spike
                'a'          : 14.,    # nS, subthreshold adaptation
                'b'          : .01,     # nA, spike-dependent adaptation
                'tau_w'      : 270.0, # ms, time constant of adaptation variable (Cerina et al. 2015)
            }
        },

        # CORTEX
        'py' : { # Regular Spiking 
            'n': 4096, # units have to be a sqrt-able to be placed in a squared (aspect ratio 1) grid 
            # 'n': 1600, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(aspect_ratio=1, dx=1.0, dy=1.0, fill_order='sequential', rng=sim.NumpyRNG(seed=138876253)),
            'cellparams': {
                'cm'         : 0.15,  # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/111/
                'tau_refrac' : 2.,    # ms, refractory period ()
                'delta_T'    : 0.8,   # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'v_thresh'   : -52.0, # mV, fixed spike threshold (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107)
                # Slow Waves - No ACh
                'tau_m'      : 17.0,  # ms, time constant of leak conductance (cm/gl, gl=8nS)
                'v_rest'     : -78.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : -67.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 1.0,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : .01,  # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
                'tau_syn_E'  : 3.,    # ms, YgerBoustaniDestexheFregnac2011
                'tau_syn_I'  : 10.0,   # ms, YgerBoustaniDestexheFregnac2011
                # Asynchronous Irregular - ACh
                # ...
                # 'tau_syn_E'  : 3.,    # ms
                # 'tau_syn_I'  : 10.0,   # ms
            }
        },
        'inh' : { # Fast Spiking 
            'n': {'ref':'py','ratio':0.25},
            'type': sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(aspect_ratio=1, dx=3.0, dy=3.0, fill_order='sequential', rng=sim.NumpyRNG(seed=138876253)),
            'cellparams': {
                'cm'         : 0.059, # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/106)
                'tau_refrac' : 2.,    # ms, refractory period ()
                'delta_T'    : 3.,    # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'v_thresh'   : -50.0, # mV, fixed spike threshold (fix McCormickPrince1986, Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107, 106)
                # Slow Waves - No ACh
                'tau_m'      : 5.0,   # ms, time constant of leak conductance (cm/gl, gl=11.8nS) s=F/S 10-9/10-9
                'v_rest'     : -60.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : -80.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 0.5,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : 0.01,  # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 28.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
                'tau_syn_E'  : 3.,    # ms, YgerBoustaniDestexheFregnac2011
                'tau_syn_I'  : 7.0,   # ms, YgerBoustaniDestexheFregnac2011
                # Asynchronous Irregular - ACh
                # ...
                # 'tau_syn_E'  : 3.,    # ms
                # 'tau_syn_I'  : 10.0,   # ms
            }
        },
    },


    'Projections' : {

        # THALAMIC
        'ext_tc' : {
            'source' : 'ext',
            'target' : 'tc',
            'receptor_type' : 'excitatory',
            'synapse_type' : sim.StaticSynapse,
            'connector' : sim.FixedProbabilityConnector(.2, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'weight' : .1, # µS
            # 'delay' : 1., 
        },
        'tc_re' : {
            'source' : 'tc',
            'target' : 're',
            'receptor_type' : 'excitatory',
            'synapse_type' : sim.StaticSynapse,
            'connector' : sim.DistanceDependentProbabilityConnector("d<2", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'weight' : .01, # µS
            # 'delay' : 1., 
        },
        're_tc' : {
            'source' : 're',
            'target' : 'tc',
            'receptor_type' : 'inhibitory',
            'synapse_type' : sim.StaticSynapse,
            'connector' : sim.DistanceDependentProbabilityConnector("d<3", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'weight' : .02, # µS (SohalPangratzRudolphHuguenard2006, SanchezMcCormick1997, LamSherman2011)
            # 'delay' : 1., 
        },
        're_re' : {
            'source' : 're',
            'target' : 're',
            'receptor_type' : 'inhibitory',
            'synapse_type' : sim.StaticSynapse,
            'connector' : sim.DistanceDependentProbabilityConnector("d<3", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'weight' : .001, # µS
            # 'delay' : 1., 
        },

        # THALAMOCORTICAL
        'tc_py' : {
            'source' : 'tc',
            'target' : 'py',
            'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.8*d)", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .0005,
            'receptor_type' : 'excitatory'
        },
        'tc_inh' : {
            'source' : 'tc',
            'target' : 'inh',
            'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.8*d)", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .001,
            'receptor_type' : 'excitatory'
        },

        # CORTICAL
        'ext_py' : {
            'source' : 'ext',
            'target' : 'py',
            'connector' : sim.FixedProbabilityConnector(.02),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .1, # µS
            # no need for delay
            'receptor_type' : 'excitatory'
        },
        'py_py' : {
            'source' : 'py',
            'target' : 'py',
            'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.3*d)", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .0013, # µS
            'delay' : .2, # ms, YgerBoustaniDestexheFregnac2011
            'receptor_type' : 'excitatory'
        },
        'py_inh' : {
            'source' : 'py',
            'target' : 'inh',
            'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.3*d)", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .0019, # µS
            'delay' : .2, # ms, 
            'receptor_type' : 'excitatory'
        },
        'inh_py' : {
            'source' : 'inh',
            'target' : 'py',
            'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.5*d)", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .02, # µS
            'delay' : .2, # ms, 
            'receptor_type' : 'inhibitory'
        },
        'inh_inh' : {
            'source' : 'inh',
            'target' : 'inh',
            'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.5*d)", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .01, # µS
            'delay' : .2, # ms, 
            'receptor_type' : 'inhibitory'
        },

        # CORTICOTHALAMIC
        'py_tc' : {
            'source' : 'py',
            'target' : 'tc',
            'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.8*d)", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .0015,
            'receptor_type' : 'excitatory'
        },
        'py_re' : {
            'source' : 'py',
            'target' : 're',
            'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.5*d)", allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'synapse_type' : sim.StaticSynapse,
            'weight' : .001,
            'receptor_type' : 'excitatory'
        },

    },


    'Recorders' : {
        'tc' : {
            'spikes' : 'all',
        },
        're' : {
            'spikes' : 'all',
        },
        'py' : {
            'spikes' : 'all',
            # 'v' : {
            #     'MUA': True,
            #     'x': 15,
            #     'y': 15,
            #     'size': 30,
            # },
        },
        'inh' : {
            'spikes' : 'all',
            # 'v' : {
            #     'random': True,
            #     'sample': 200,
            # },
            # 'v' : {
            #     'start' : 100,
            #     'end' : 110,
            # }
        },

    },


    'Modifiers' :{
    },


    'Injections' : {
    },


    'Analysis' : {
        'Static' : False,
        'Vm' : False,
        'PhaseSpace' : False,
        'Rasterplot' : False,
        # 'ISI' : {
        #     'Populations': ['py','tc'],
        #     'bin_size': 100, # * dt: 100 * 0.1 = 10 ms
        # },
        'ISI' : False,
        'ISI#' : False,
        'CrossCorrelation' : False,
        'FiringRate' : True,
        'LFP' : False,
    },

}