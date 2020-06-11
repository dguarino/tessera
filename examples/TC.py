# ------------------------------------------------------------------------------
# Usage:
# Examples

# start the docker image:
# $ docker run -v `pwd`:`pwd` -w `pwd` -i -t tessera /bin/bash

# Run simple code
# python run.py --folder test --params TC_response.py nest

# Search Example:
# python run.py --folder EPSPsearch --params TC_response.py --search search.py --map yes nest
# python plot_map.py

# Analysis Example
# python run.py --folder EPSPsearch --params TC_response.py --search search.py --analysis true nest

# ------------------------------------------------------------------------------

{

    'run_time': 1000., # ms
    'dt': 0.01, # ms

    'Populations' : {

        ##################################
        # 1 neuron population to simulate ACh effects on short-term post-synaptic plasticity (PSP)
        # 'ext' : {
        #     'n' : 1,
        #     # 'type': sim.SpikeSourceArray(spike_times=np.arange(200.,325.,4.7)), # ~210 Hz, KimMcCormick1998 
        #     # 'type': sim.SpikeSourceArray(spike_times=np.arange(200.,325.,6.2)), # ~160 Hz, KimMcCormick1998 
        #     # 'type': sim.SpikeSourceArray(spike_times=np.arange(200.,325.,12.5)), # 80 Hz, KimMcCormick1998 
        #     'type': sim.SpikeSourceArray(spike_times=[200., 225., 250., 275., 300., 325., 350., 375.]), # 40 Hz, KimMcCormick1998
        #     'cellparams': None,
        # },

        'cell' : {
            'n': 1, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            'cellparams': {
                'cm'         : 0.16,  # nF, tot membrane capacitance (Bloomfield Hamos Sherman 1987)
                'tau_refrac' : 2.5,   # ms, refractory period (ReinagelReid2000)
                'delta_T'    : 4.5,   # mV, steepness of exponential approach to threshold (Destexhe2009)
                'v_spike'    : 20.0,  # mV, spike detection
                'v_thresh'   : -50.0, # mV, fixed spike threshold (https://www.neuroelectro.org/neuron/190/)
                # 'v_thresh'   : RandomDistribution('normal', (-52.0, 2.0), rng=NumpyRNG(seed=85524)),

                ##################################
                # ACh effects on input resistance

                # With ACh
                'tau_m'      : 16.,   # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                'v_rest'     : -55.0, # mV, resting potential (McCormick Pape 1988)
                'v_reset'    : -50.0, # mV, reset after spike
                'a'          : .0,    # nS, spike-frequency adaptation
                'b'          : .01,   # nA, increment to the adaptation variable
                'tau_w'      : 200.0, # ms, time constant of adaptation variable

                # # Middle ACh
                # 'tau_m'      : 17.,  # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                # 'v_rest'     : -63.0, # mV, resting potential
                # 'v_reset'    : -48.0, # mV, reset after spike
                # 'a'          : 14.,    # nS, spike-frequency adaptation
                # 'b'          : .02,   # nA, increment to the adaptation variable
                # 'tau_w'      : 270.0, # ms, time constant of adaptation variable

                # # Whitout ACh
                # 'tau_m'      : 18.,   # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                # 'v_rest'     : -75.0, # mV, resting potential (McCormick Pape 1988)
                # 'v_reset'    : -46.0, # mV, reset after spike
                # 'a'          : 28.,   # nS, spike-frequency adaptation
                # 'b'          : .02,   # nA, increment to the adaptation variable
                # 'tau_w'      : 270.0, # ms, time constant of adaptation variable
           
                ##################################
                # Cell level time constants of short-term post-synaptic plasticity (PSP)
                'tau_syn_E'  : 5.,    #  control and ACh # ms, time constant of excitatory synaptic short-term plasticity
                'tau_syn_I'  : 10.0,  # ms, time constant of inhibitory synaptic short-term plasticity
           }
        },
    },

    'Projections' : {
        # # Periphery -> TC
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=1., tau_rec=30.0, tau_facil=21.0),
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     'delay' : .01, # ms  
        # },    

        # # RS -> TC
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=1., tau_rec=30.0, tau_facil=21.0),
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     'delay' : .01, # ms  
        # },    

        # # RE -> TC
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'inhibitory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=1., tau_rec=30.0, tau_facil=21.0),
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     'delay' : .01, # ms  
        # },    
    },

    'Injections' : {
        'cell' : {
            'source' : sim.StepCurrentSource,
            ##################################
            # ACh effects on input resistance

            # protocol for an awake resting -63.0, see above refs
            # 'amplitude' : [.2, .0], # no result
            # 'amplitude' : [.3, .0], # initial burst, then silent
            # 'amplitude' : [.37, .0], # postburst subthreshold oscillatory dynamic
            # 'amplitude' : [.1, .0], # chattering at ~5Hz
            # 'amplitude' : [.5, .0], # tonic
            # 'amplitude' : [.36, .0], # 

            # protocol as in figure 2C AvanziniCurtisPanzicaSpreafico1989
            # 'amplitude' : [.10, .0], # no result
            # 'amplitude' : [.55, .0], # initial burst, then silent
            # 'amplitude' : [.58, .0], # 
            # 'amplitude' : [.95, .0], # postburst subthreshold oscillatory dynamic
            # 'amplitude' : [1., .0], # chattering at ~1Hz
            # 'amplitude' : [1.1, .0], # tonic

            'amplitude' : [.35, .0], # default
            # 'start' : [200., 350.], # McCormick Pape
            'start' : [100., 900.], # short duration
            'stop' : 0.0
        },
    },

    'Recorders' : {
        'cell' : {
            'spikes' :  'all',
            'v' : 'all',
            'w' : 'all',
            #'v' : {
            #    'start':0,
            #    'end':1
            #}
        },
    },

    'Modifiers' :{
    },


    'Analysis' : {
        'Static' : False,
        'Vm' : True,
        'PhaseSpace' : True,
        'Rasterplot' : False,
        'ISI' : False,
        'ISI#' : False,
        'CrossCorrelation' : False,
        'FiringRate' : False,
        'LFP' : False,
    },

}
