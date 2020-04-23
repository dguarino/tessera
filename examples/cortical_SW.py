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
    'run_time': 20000, # ms, 20 sec
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
            'n': 64*64, # units have to be placed in a squared (aspect ratio 1) grid 
            # 'n': 1600, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            # 'structure' : Grid2D(aspect_ratio=1, dx=1.0, dy=1.0, fill_order='sequential'),
            'structure' : Grid2D(aspect_ratio=1, dx=1.0, dy=1.0, fill_order='random', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams': {
                'cm'         : 0.15,  # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/111/
                'tau_refrac' : 2.,    # ms, refractory period ()
                'delta_T'    : 0.8,   # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'v_thresh'   : -52.0, # mV, fixed spike threshold (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107)
                # Slow Waves - No ACh
                'tau_m'      : 17.0,  # ms, time constant of leak conductance (cm/gl, gl=8nS)
                # 'v_rest'     : -75.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -70.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'v_rest'     : simrand.RandomDistribution('normal', mu=-75., sigma=0.01), # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : simrand.RandomDistribution('normal', mu=-70., sigma=0.01), # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 1.0,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : .01,  # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
                'tau_syn_E'  : 3.,    # ms, 
                'tau_syn_I'  : 10.0,   # ms, 
                # Asynchronous Irregular - ACh
                # ...
                # 'tau_syn_E'  : 3.,    # ms
                # 'tau_syn_I'  : 10.0,   # ms
            }
        },
        'inh' : { # Fast Spiking 
            'n': {'ref':'py','ratio':0.25},
            'type': sim.EIF_cond_alpha_isfa_ista,
            # 'structure' : Grid2D(aspect_ratio=1, dx=2.0, dy=2.0, fill_order='sequential'),
            'structure' : Grid2D(aspect_ratio=1, dx=2.0, dy=2.0, fill_order='random', rng=sim.NumpyRNG(seed=2**32-1)),
            'cellparams': {
                'cm'         : 0.059, # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/106)
                'tau_refrac' : 2.,    # ms, refractory period ()
                'delta_T'    : 3.,    # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'v_thresh'   : -50.0, # mV, fixed spike threshold (fix McCormickPrince1986, Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107, 106)
                # Slow Waves - No ACh
                'tau_m'      : 5.0,   # ms, time constant of leak conductance (cm/gl, gl=11.8nS) s=F/S 10-9/10-9
                # 'v_rest'     : -56.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -74.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'v_rest'     : simrand.RandomDistribution('normal', mu=-56., sigma=0.01), # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : simrand.RandomDistribution('normal', mu=-74., sigma=0.01), # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 0.5,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : 0.01,  # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 28.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
                'tau_syn_E'  : 5.,    # ms
                'tau_syn_I'  : 10.,   # ms
            }
        },
    },


    'Projections' : {

        'ext_py' : {
            'source' : 'ext',
            'target' : 'py',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.FixedProbabilityConnector(.02, rng=sim.NumpyRNG(2**32-1)),
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
        # 'py_py' : {
        #     'source' : 'py',
        #     'target' : 'py',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=.4, tau_rec=400.0, tau_facil=0.0), # Tsodyks and Markram 1997 Control
        #     # 'synapse_type' : sim.TsodyksMarkramSynapse(U=0.2, tau_rec=1900.0, tau_facil=0.0), # Tsodyks and Markram 1997 ACh
        #     'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.3*d)", allow_self_connections=False),
        #     'weight' : .0013, # µS
        #     'delay' : .2, # ms
        # },

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
        # 'py_inh' : {
        #     'source' : 'py',
        #     'target' : 'inh',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=.5, tau_rec=400., tau_facil=0.0), # Levy et al. 2008 Control
        #     # 'synapse_type' : sim.TsodyksMarkramSynapse(U=.4, tau_rec=50., tau_facil=0.0), # Levy et al. 2008 ACh
        #     'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.3*d)", allow_self_connections=False),
        #     'weight' : 0.0019, # µS
        #     'delay' : 0.2, # ms
        # },

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
        # 'inh_py' : {
        #     'source' : 'inh',
        #     'target' : 'py',
        #     'receptor_type' : 'inhibitory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=.25, tau_rec=20., tau_facil=0.0), # Beierlein et al. 2003 Control
        #     # 'synapse_type' : sim.TsodyksMarkramSynapse(U=.4, tau_rec=420., tau_facil=0.0), # GigoutJonesWierschkeDaviesWatsonDeisz2012
        #     'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.5*d)", allow_self_connections=False),
        #     'weight' : 0.02, # µS
        #     'delay' : .2, # ms
        # },

        'inh_inh' : {
            'source' : 'inh',
            'target' : 'inh',
            'space' :  sim.Space(periodic_boundaries=((0,64), (0,64), None)), # torus
            'connector' : sim.DistanceDependentProbabilityConnector("14*exp(-2*d)", allow_self_connections=False, rng=sim.NumpyRNG(2**32-1)), # -0.3 results in the 0.1 probability of connecting at 4.5 grid distance
            'synapse_type' : sim.StaticSynapse(),
            'weight' : .0025, # µS
            'delay' : .5, # ms, 
            'receptor_type' : 'inhibitory'
        },
        # 'inh_inh' : {
        #     'source' : 'inh',
        #     'target' : 'inh',
        #     'receptor_type' : 'inhibitory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=.5, tau_rec=40., tau_facil=0.0), # Control
        #     # 'synapse_type' : sim.TsodyksMarkramSynapse(U=.2, tau_rec=40., tau_facil=0.0), # ACh
        #     'connector' : sim.DistanceDependentProbabilityConnector("exp(-0.5*d)", allow_self_connections=False),
        #     'weight' : 0.01, # µS
        #     'delay' : 0.2, # ms
        # },

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
        'Vm' : False,
        'PhaseSpace' : False,
        'Static' : False,
    },

}