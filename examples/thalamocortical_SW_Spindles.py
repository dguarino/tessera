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
    'run_time': 30000, # ms
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
            'n': 16*16, # Npy * 0.0625 (Destexhe)
            'type' :  sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(dx=1.0, dy=1.0, fill_order='random', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams' : {
                'tau_syn_E'  : 3.0,   # ms
                'tau_syn_I'  : 7.0,   # ms
                'tau_refrac' : 2.5,   # ms, refractory period (ReinagelReid2000)
                'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold (Destexhe2009)
                'v_thresh'   : -50.0, # mV, fixed spike threshold (https://www.neuroelectro.org/neuron/190/)
                'cm'         : 0.16,  # nF, tot membrane capacitance (Bloomfield Hamos Sherman 1987)
                # Spindles
                'tau_m'      : 17.,  # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                'v_rest'     : -63.0, # mV, resting potential
                'v_reset'    : -48.0, # mV, reset after spike
                'a'          : 26.,    # nS, spike-frequency adaptation
                'b'          : .02,   # nA, increment to the adaptation variable
                'tau_w'      : 270.0, # ms, time constant of adaptation variable
            }
        },
        're' : {
            'n': 16*16, # Npy * 0.0625 (Destexhe)
            'type' :  sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(dx=1.0, dy=1.0, fill_order='random', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams' : {
                'tau_syn_E'  : 3.0,   # ms
                'tau_syn_I'  : 7.0,   # ms
                'tau_refrac' : 2.5,   # ms, refractory period (ReinagelReid2000)
                'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold (Destexhe2009)
                'v_thresh'   : -50.0, # mV, fixed spike threshold (https://www.neuroelectro.org/neuron/190/)
                'cm'         : 0.20,  # nF, tot membrane capacitance (https://www.neuroelectro.org/neuron/190/, Uhlrich Cucchiaro Humphrey Sherman 1991)
                # Spindles
                'tau_m'      : 15.,   # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                'v_rest'     : -70.0, # mV, resting potential
                'v_reset'    : -41.0, # mV, reset after spike (McCormick1992, Toubul and Brette 2008)
                'a'          : 28.,   # nS, 
                'b'          : .02,   # nA, spike-dependent adaptation
                'tau_w'      : 270.0, # ms, time constant of adaptation variable
            }
        },

        # CORTEX
        'py' : { # Regular Spiking 
            'n': 64*64, # units have to be placed in a squared (aspect ratio 1) grid 
            'type': sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(aspect_ratio=1, dx=1.0, dy=1.0, fill_order='random', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams': {
                'tau_syn_E'  : 3.,    # ms, 
                'tau_syn_I'  : 10.0,   # ms, 
                'cm'         : 0.15,  # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/111/
                'tau_refrac' : 2.,    # ms, refractory period ()
                'delta_T'    : 0.8,   # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'v_thresh'   : -52.0, # mV, fixed spike threshold (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107)
                # Slow Waves - No ACh
                'tau_m'      : 17.0,  # ms, time constant of leak conductance (cm/gl, gl=8nS)
                'v_rest'     : -75.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : -70.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'v_rest'     : simrand.RandomDistribution('normal', mu=-75., sigma=0.01), # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : simrand.RandomDistribution('normal', mu=-70., sigma=0.01), # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 1.0,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : .01,  # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
            }
        },
        'inh' : { # Fast Spiking 
            'n': {'ref':'py','ratio':0.25},
            'type': sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(aspect_ratio=1, dx=2.0, dy=2.0, fill_order='random', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams': {
                'tau_syn_E'  : 5.,    # ms
                'tau_syn_I'  : 10.,   # ms
                'cm'         : 0.059, # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/106)
                'tau_refrac' : 2.,    # ms, refractory period ()
                'delta_T'    : 3.,    # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'v_thresh'   : -50.0, # mV, fixed spike threshold (fix McCormickPrince1986, Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107, 106)
                # Slow Waves - No ACh
                'tau_m'      : 5.0,   # ms, time constant of leak conductance (cm/gl, gl=11.8nS) s=F/S 10-9/10-9
                'v_rest'     : -56.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : -74.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'v_rest'     : simrand.RandomDistribution('normal', mu=-56., sigma=0.01), # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : simrand.RandomDistribution('normal', mu=-74., sigma=0.01), # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 0.5,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : 0.01,  # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 28.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
            }
        },

    },


    'Projections' : {

        # THALAMIC
        'ext_tc' : {
            'source' : 'ext',
            'target' : 'tc',
            'space' :  sim.Space(periodic_boundaries=((0,16), (0,16), None)), # torus
            'receptor_type' : 'excitatory',
            'synapse_type' : sim.StaticSynapse(),
            'connector' : sim.FixedProbabilityConnector(.2, allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'weight' : .1, # µS
            'delay' : .2, 
        },
        'tc_re' : {
            'source' : 'tc',
            'target' : 're',
            'space' :  sim.Space(periodic_boundaries=((0,16), (0,16), None)), # torus
            'receptor_type' : 'excitatory',
            'synapse_type' : sim.StaticSynapse(),
            'connector' : sim.DistanceDependentProbabilityConnector("d<2", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'weight' : .01, # µS
            'delay' : .2, 
        },
        're_tc' : {
            'source' : 're',
            'target' : 'tc',
            'space' :  sim.Space(periodic_boundaries=((0,16), (0,16), None)), # torus
            'receptor_type' : 'inhibitory',
            'synapse_type' : sim.StaticSynapse(),
            'connector' : sim.DistanceDependentProbabilityConnector("d<3", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'weight' : .02, # µS (SohalPangratzRudolphHuguenard2006, SanchezMcCormick1997, LamSherman2011)
            'delay' : .2, 
        },        
        're_re' : {
            'source' : 're',
            'target' : 're',
            'space' :  sim.Space(periodic_boundaries=((0,16), (0,16), None)), # torus
            'receptor_type' : 'inhibitory',
            'synapse_type' : sim.StaticSynapse(),
            'connector' : sim.DistanceDependentProbabilityConnector("d<3", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'weight' : .001, # µS
            'delay' : .2, 
        },

        # THALAMOCORTICAL
        'tc_py' : {
            'source' : 'tc',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-3*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .004,
            'delay' : 2., # ms, 
            'receptor_type' : 'excitatory'
        },
        'tc_inh' : {
            'source' : 'tc',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-3*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .004,
            'delay' : 2., # ms, 
            'receptor_type' : 'excitatory'
        },

        # CORTICAL
        'ext_py' : {
            'source' : 'ext',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .1, # µS
            # no need for delay
            'receptor_type' : 'excitatory'
        },
        'py_py' : {
            'source' : 'py',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)), # -0.3 results in the 0.1 probability of connecting at 7.5 grid distance
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0025, # µS
            'delay' : .5, # ms, YgerBoustaniDestexheFregnac2011
            'receptor_type' : 'excitatory'
        },
        'py_inh' : {
            'source' : 'py',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)), # -0.3 results in the 0.1 probability of connecting at 7.5 grid distance
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0015, # µS
            'delay' : .5, # ms, 
            'receptor_type' : 'excitatory'
        },
        'inh_py' : {
            'source' : 'inh',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)), # -0.3 results in the 0.1 probability of connecting at 4.5 grid distance
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0015, # µS
            'delay' : .5, # ms, 
            'receptor_type' : 'inhibitory'
        },
        'inh_inh' : {
            'source' : 'inh',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)), # -0.3 results in the 0.1 probability of connecting at 4.5 grid distance
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0015, # µS
            'delay' : .5, # ms, 
            'receptor_type' : 'inhibitory'
        },

        # CORTICOTHALAMIC
        'py_tc' : {
            'source' : 'py',
            'target' : 'tc',
            'space' :  sim.Space(periodic_boundaries=((0,16), (0,16), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-1.5*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0025,
            'delay' : 2., # ms, 
            'receptor_type' : 'excitatory'
        },
        'py_re' : {
            'source' : 'py',
            'target' : 're',
            'space' :  sim.Space(periodic_boundaries=((0,16), (0,16), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-1.5*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .001,
            'delay' : 2., # ms, 
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
        # 'Coherence': {
        #     'Population1': 'py',
        #     # 'Population2': 're',
        #     'Population2': 'tc',
        # },
        # 'Movie' : {
        #     'populations': {
        #         'py' : {
        #             'plot': True,
        #             'ratio': 1,
        #             'color': 'red',
        #         },
        #         'inh' : {
        #             'plot': False,
        #             'ratio': 4,
        #             'color': 'blue',
        #         },
        #     },
        #     'from': 20000,
        #     'to': 21000,
        # },
        'Rasterplot' : {
            'tc':{
                'limits': [(0,16),(0,16)], # all
            },
            're':{
                'limits': [(0,16),(0,16)], # all
            },
            'py':{
                'limits': [(10,50),(10,50)], # only central ones
            },
            'inh':{
                'limits': [(10,50),(10,50)], # only central ones
            },
            'type': '.png',
            # 'type': '.svg',
            'interval': False, # all
            # 'interval': [2000.,3000.], # ms # from 2s to 3s
            'dpi':800,
        },
        # 'Autocorrelation' : {
        #     'populations': ['py','tc'],
        #     # 'populations': ['py','inh','inh','re'],
        #     'bin_size': 100, # * dt: 100 * 0.1 = 10 ms
        # },
        # 'ISI' : {
        #     'py':{
        #         'bin': 50, # ms, 20 per second
        #         'limits': [(16,16),(48,48)], # cells to be considered in the analysis (to avoid bounduary effects)
        #     },
        # },
        'FiringRate' : {
            'tc':{
                'firing': [0,200],
            },
            're':{
                'firing': [0,200],
            },
            'py':{
                'firing': [0,200],
            },
            'inh':{
                'firing': [0,200],
            },
        },
        'Vm' : False,
        'ISI#' : False,
        'CrossCorrelation' : False,
        'LFP' : False,
        'PhaseSpace' : False,
        'Static' : False,
    },

}