{
    'run_time': 4000, # ms
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

        'py' : { # Regular Spiking 
            'n': 64*64, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(aspect_ratio=1, dx=1.0, dy=1.0, fill_order='random', rng=sim.NumpyRNG(seed=2**32-1)), 
            'cellparams': {
                'tau_syn_E'  : 3.,    # ms, YgerBoustaniDestexheFregnac2011
                'tau_syn_I'  : 7.0,   # ms, YgerBoustaniDestexheFregnac2011
                'tau_refrac' : 2.,    # ms, refractory period ()
                'v_thresh'   : -52.0, # mV, fixed spike threshold (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107)
                'delta_T'    : 0.8,   # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'cm'         : 0.15,  # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/111/, YgerBoustaniDestexheFregnac2011)
                # local
                'tau_m'      : 30.0,  # ms, time constant of leak conductance (cm/gl, gl=5nS)
                'v_rest'     : -65.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : -67.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 1.0,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : 0.005, # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
                # # ACh @resting -52mV (McCormickPrince1986 Fig. 3A holding -60mV)
                # 'tau_m'      : 30.0,  # ms, time constant of leak conductance (cm/gl, gl=5nS)
                # 'v_rest'     : -58.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -67.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'a'          : 1.0,   # nS, conductance of adaptation variable (Naud et al. 2008)
                # 'b'          : 0.005, # nA, increment to the adaptation variable (Naud et al. 2008)
                # 'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
                # # ACh @resting -75mV (McCormickPrince1986 Fig. 3B holding -80mV)
                # 'tau_m'      : 20,  # ms, time constant of leak conductance (cm/gl, gl=7.5nS) PRE
                # #'tau_m'      : 25,  # ms, time constant of leak conductance (cm/gl, gl=6nS) POST1
                # 'v_rest'     : -75.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -55.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'a'          : .1,   # nS, conductance of adaptation variable (Naud et al. 2008)
                # 'b'          : 0.1, # nA, increment to the adaptation variable (Naud et al. 2008)
                # 'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)

            }
        },
        'inh' : { # Fast Spiking
            'n': {'ref':'py','ratio':0.25},
            'type': sim.EIF_cond_alpha_isfa_ista,
            'structure' : Grid2D(aspect_ratio=1, dx=1.0, dy=1.0, fill_order='random', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams': {
                'tau_syn_E'  : 3.,    # ms, YgerBoustaniDestexheFregnac2011
                'tau_syn_I'  : 7.0,   # ms, YgerBoustaniDestexheFregnac2011
                'tau_refrac' : 2.,    # ms, refractory period ()
                'cm'         : 0.05,  # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/106)
                'v_thresh'   : -42.0, # mV, fixed spike threshold (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107, 106)
                'delta_T'    : 3.,    # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'tau_m'      : 5.0,  # ms, time constant of leak conductance (cm/gl, gl=11.8nS) s=F/S 10-9/10-9
                'v_rest'     : -41.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : -45.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 0.8,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : 0.081, # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 28.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
            }
        },
    },


    'Projections' : {
        'ext_py' : {
            'source' : 'ext',
            'target' : 'py',
            'connector' : sim.FixedProbabilityConnector(.02),
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .03, # µS
            'receptor_type' : 'excitatory'
        },
        'py_py' : {
            'source' : 'py',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            #'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.random.NumpyRNG(1235342134, parallel_safe=False)),
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)), #
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0015, # µS
            'delay' : .2, # ms, YgerBoustaniDestexheFregnac2011
            'receptor_type' : 'excitatory'
        },
        'py_inh' : {
            'source' : 'py',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            #'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.random.NumpyRNG(1235342134, parallel_safe=False)),
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)), #
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0015, # µS
            'delay' : .2, # ms, YgerBoustaniDestexheFregnac2011
            'receptor_type' : 'excitatory'
        },
        'inh_py' : {
            'source' : 'inh',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            #'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.random.NumpyRNG(1235342134, parallel_safe=False)),
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)),
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0015, # µS
            'delay' : .2, # ms, YgerBoustaniDestexheFregnac2011
            'receptor_type' : 'inhibitory'
        },
        'inh_inh' : {
            'source' : 'inh',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            #'connector' : sim.FixedProbabilityConnector(.02, allow_self_connections=False, rng=sim.random.NumpyRNG(1235342134, parallel_safe=False)),
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)), #
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0015, # µS
            'delay' : .2, # ms, YgerBoustaniDestexheFregnac2011
            'receptor_type' : 'inhibitory'
        }
    },

    'Recorders' : {
        'py' : {
            'spikes' : 'all',
            # 'v' : {
            #     'MUA': True,
            #     'x': 15,
            #     'y': 15,
            #     'size': 30,
            # },
            'v' : {
                'start' : 1000,
                'end' : 1010,
            }
        },
        'inh' : {
            'spikes' : 'all',
            # 'v' : {
            #     'random': True,
            #     'sample': 200,
            # },
            'v' : {
                'start' : 1000,
                'end' : 1010,
            }
        },

    },


    'Modifiers' :{
        # 'py' : {
        # 'cells' : {
        #        'start' : 0,
        #        'end' : 0.01
        #     },
        #     'properties' : {
        #         'tau_w' : 150.,
        #         'cm' : 0.15,
        #         'tau_m' : 30.0, #
        #         'a' : 12., #Alain 0.02, #uS
        #         'b' : .03 #0.0
        #     }
        # }
    },


    'Injections' : {
    },


    'Analysis' : {
        # 'Coherence': {
        #     'Population1': 'py',
        #     'Population2': 'inh',
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
            'py':{
                # 'limits': [(0,63),(0,63)], # coords: [(from x, to x), (from y, to y)] 
                'limits': [(10,50),(10,50)], # only central ones
            },
            'inh':{
                # 'limits': [(0,63),(0,63)], # coords: [(from x, to x), (from y, to y)] 
                'limits': [(10,50),(10,50)], # only central ones
            },
            'type': '.png',
            # 'type': '.svg',
            'interval': False, # all
            # 'interval': [2000.,3000.], # ms # from 2s to 3s
            'dpi':800,
        },
        # 'ISI' : {
        #     'py':{
        #         'bin': 50, # ms, 20 per second
        #         # 'limits': [(0,63),(0,63)], # coords: [(from x, to x), (from y, to y)] 
        #         'limits': [(10,50),(10,50)], # only central ones
        #     },
        # },
        'FiringRate' : {
            'py':{
                'firing': [0,200],
            },
            'inh':{
                'firing': [0,200],
            },
        },
        'ISI#' : False,
        'CrossCorrelation' : False,
        'LFP' : False,
        'Vm' : True,
        'PhaseSpace' : False,
        'Static' : False,
    },

}

