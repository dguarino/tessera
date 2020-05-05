# ------------------------------------------------------------------------------
# Usage:
# Examples

# start the docker image (mapping current directory into the container):
# $ docker run -v `pwd`:`pwd` -w `pwd` -i -t tessera /bin/bash

# Run one simulation
# python run.py --folder test --params thalamic_delta.py nest

# Run parameter search
# python run.py --folder test --params thalamic_delta.py --search search.py nest

# Analysis only
# python run.py --folder test --params thalamic_delta.py --analysis true nest

# ------------------------------------------------------------------------------
{
    'run_time': 5000, # ms
    'dt': 0.1, # ms

    'Populations' : {

        'ext' : {
            'n' : 1,
            'type': sim.SpikeSourcePoisson,
            'cellparams' : {
                'start':0.0,
                'rate':50.,
                'duration':100.0 # initial kick
            }
        },

        'tc' : {
            'n': 100,
            'type' :  sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(dx=1.0, dy=1.0, fill_order='random'),#, rng=sim.NumpyRNG(seed=13886)),
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
            'structure' : Grid2D(dx=1.0, dy=1.0, fill_order='random'),#, rng=sim.NumpyRNG(seed=13886)),
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
        }

    },


    'Projections' : {

        'ext_tc' : {
            'source' : 'ext',
            'target' : 'tc',
            'space' :  sim.Space(periodic_boundaries=((0,10), (0,10), None)), # torus
            'receptor_type' : 'excitatory',
            'synapse_type' : sim.StaticSynapse(),
            'connector' : sim.FixedProbabilityConnector(.2, allow_self_connections=False, rng=sim.NumpyRNG(1235342134)),
            'weight' : .1, # µS
            'delay' : .2, 
        },

        'tc_re' : {
            'source' : 'tc',
            'target' : 're',
            'space' :  sim.Space(periodic_boundaries=((0,10), (0,10), None)), # torus
            'receptor_type' : 'excitatory',
            'synapse_type' : sim.StaticSynapse(),
            # 'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'connector' : sim.DistanceDependentProbabilityConnector("d<2", allow_self_connections=False, rng=sim.NumpyRNG(1235342134)),
            'weight' : .01, # µS
            'delay' : .2, 
        },
        're_tc' : {
            'source' : 're',
            'target' : 'tc',
            'space' :  sim.Space(periodic_boundaries=((0,10), (0,10), None)), # torus
            'receptor_type' : 'inhibitory',
            'synapse_type' : sim.StaticSynapse(),
            # 'connector' : sim.FixedProbabilityConnector(.08, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'connector' : sim.DistanceDependentProbabilityConnector("d<3", allow_self_connections=False, rng=sim.NumpyRNG(1235342134)),
            'weight' : .02, # µS (SohalPangratzRudolphHuguenard2006, SanchezMcCormick1997, LamSherman2011)
            'delay' : .2, 
        },
        
        're_re' : {
            'source' : 're',
            'target' : 're',
            'space' :  sim.Space(periodic_boundaries=((0,10), (0,10), None)), # torus
            'receptor_type' : 'inhibitory',
            'synapse_type' : sim.StaticSynapse(),
            # 'connector' : sim.FixedProbabilityConnector(.08, allow_self_connections=False, rng=sim.NumpyRNG(1235342134, parallel_safe=False)),
            'connector' : sim.DistanceDependentProbabilityConnector("d<3", allow_self_connections=False, rng=sim.NumpyRNG(1235342134)),
            'weight' : .001, # µS
            'delay' : .2, 
        },
    
    },


    'Recorders' : {
        'tc' : {
            'spikes' :  'all',
            # 'gsyn_exc' : 'all',
            # 'v' : 'all',
            'v' : {
                'start' : 0,
                'end' : 10,
            },
        },
        're' : {
            'spikes' :  'all',
            # 'gsyn_exc' : 'all',
            # 'v' : 'all',
            'v' : {
                'start' : 0,
                'end' : 10,
            },
        }
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
                'limits': [(0,10),(0,10)], # all
            },
            're':{
                'limits': [(0,10),(0,10)], # all
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
        },
        'Vm' : False,
        'ISI#' : False,
        'CrossCorrelation' : False,
        'LFP' : False,
        'PhaseSpace' : False,
        'Static' : False,
    },


    'Modifiers' :{
    },


    'Injections' : {
    },

}