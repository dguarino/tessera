# ------------------------------------------------------------------------------
# Usage:
# Examples

# start the docker image:
# $ docker run -v `pwd`:`pwd` -w `pwd` -i -t tessera /bin/bash

# Run simple code
# python run.py --folder test --params RE_response.py nest

# Search Example:
# python run.py --folder EPSPsearch --params RE_response.py --search search.py --map yes nest
# python plot_map.py

# Analysis only Example
# python run.py --folder EPSPsearch --params RE_response.py --search search.py --analysis true nest

# ------------------------------------------------------------------------------

{

    'run_time': 1000., # ms
    'dt': 0.01, # ms

    'Populations' : {
        ##################################
        # 1 neuron population to simulate ACh effects on short-term post-synaptic plasticity (PSP)
        'ext' : {
            'n' : 1,
            # 'type': sim.SpikeSourceArray(spike_times=np.arange(200.,300.,4.7)), # ~210 Hz, KimMcCormick1998 
            'type': sim.SpikeSourceArray(spike_times=np.arange(200.,300.,6.2)), # ~160 Hz, KimMcCormick1998 
            # 'type': sim.SpikeSourceArray(spike_times=np.arange(200.,300.,12.5)), # 80 Hz, KimMcCormick1998 
            # 'type': sim.SpikeSourceArray(spike_times=np.arange(200.,300.,25.)), # 40 Hz, KimMcCormick1998
            'cellparams': None,
        },

        'cell' : {
            'n': 1, # units
            'type': sim.EIF_cond_alpha_isfa_ista,
            'cellparams': {
                'tau_refrac' : 2.5,   # ms, refractory period (ReinagelReid2000)
                'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold (Destexhe2009)
                'v_spike'   : 20.0,   # mV, spike detection
                # 'v_thresh'   : -50.0, # mV, fixed spike threshold (https://www.neuroelectro.org/neuron/190/)
                'v_thresh'   : -45.0, # mV Mulle et al. 1986, Avanzini et al. 1989
                'cm'         : 0.20,  # nF, tot membrane capacitance (https://www.neuroelectro.org/neuron/190/, Uhlrich Cucchiaro Humphrey Sherman 1991)

                # # With ACh
                # 'tau_m'      : 20.,  # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                # 'v_rest'     : -85.0, # mV, resting potential (McCormick Prince 1986)
                # 'v_reset'    : -45.0, # mV, reset after spike
                # 'a'          : 28.,    # nS, spike-frequency adaptation
                # 'b'          : .0,     # nA, spike-dependent adaptation
                # 'tau_w'      : 200.0, # ms, time constant of adaptation variable

                # # middle ACh
                # 'tau_m'      : 15.,  # ms, time constant of leak conductance (cm/gl, with gl=0.01)
                # 'v_rest'     : -70.0, # mV, resting potential
                # 'v_reset'    : -41.0, # mV, reset after spike (McCormick1992, Toubul and Brette 2008)
                # 'a'          : 28.,   # nS, 
                # 'b'          : .01,     # nA, spike-dependent adaptation
                # 'tau_w'      : 230.0, # ms, time constant of adaptation variable

                # # Whitout ACh
                # 'tau_m'      : 6.,  # ms, time constant of leak conductance (cm/gl)
                # 'v_rest'     : -65.0, # mV, resting potential (McCormick Prince 1986)
                # 'v_reset'    : -50.0, # mV, reset after spike
                # 'a'          : 2.,    # nS, subthreshold adaptation
                # 'b'          : .01,     # nA, spike-dependent adaptation
                # 'tau_w'      : 270.0, # ms, time constant of adaptation variable (Cerina et al. 2015)

                # ##### Mulle et al. 1986, Avanzini et al. 1989
                'tau_m'      : 20.,  # ms, time constant of leak conductance (cm/gl)
                'v_rest'     : -65.0, # mV, resting potential
                'v_reset'    : -41.0, # mV, reset after spike
                'a'          : 1.,    # nS, regular repetitive bursting
                # 'a'          : 2.,    # nS, adapting repetitive bursting
                # 'a'          : -1.3,    # !nS, chaotic repetitive bursting
                'b'          : .04,     # nA, spike-dependent adaptation
                'tau_w'      : 270.0, # ms, time constant of adaptation variable (Cerina et al. 2015)
                # 
                # 'v_thresh'   : -45.0, # mV, fixed spike threshold
                # 'v_spike'    : -20.0, # mV, spike detection
                # 'a'          : .0,    # nS, subthreshold adaptation
                # 'b'          : .04,  # nA, increment to the adaptation variable
                # 'v_reset'    : -41.0, # mV, reset after spike
                # 'tau_m'      : 20.,  # ms, time constant of leak conductance (cm/gl)
           
                ##################################
                # Cell level time constants of short-term post-synaptic plasticity (PSP)
                'tau_syn_E'  : 1.,    #  control and ACh # ms, time constant of excitatory synaptic short-term plasticity
                
                'tau_syn_I'  : 10.0,  # ms, time constant of inhibitory synaptic short-term plasticity
            }
        },
    },

    'Projections' : {
        # # RS -> RE
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=1., tau_rec=30.0, tau_facil=21.0, weight=0.01, delay=0.5),
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     'delay' : 0.01, 
        # },    

        # # RE -> RE
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'inhibitory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=1., tau_rec=30.0, tau_facil=21.0, weight=0.01, delay=0.5),
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     'delay' : 0.01, 
        # },    

        # TC -> RE
        'ext_cell' : {
            'source' : 'ext',
            'target' : 'cell',
            'receptor_type' : 'excitatory',
            'synapse_type' : sim.TsodyksMarkramSynapse(U=0.15, tau_rec=8.0, tau_facil=200.0), # Control, KimMcCormick1998
            'connector' : sim.AllToAllConnector(),
            'weight' : .01, # µS
            'delay' : 0.01, 
        },    
    },

    'Injections' : {
        # 'cell' : {
        #     'source' : sim.StepCurrentSource,

        #     # NOTES: 
        #     # - the model reproduces Avanzini et al. at -90, and it does the same with CurooDossi et al. at -63, more reactive in a smaller dynamic scale
        #     # - the amplitude of injection is encoded in the oscillatory freq

        #     # protocol for an awake resting -63.0, see above refs
        #     # 'amplitude' : [.2, .0], # no result
        #     # 'amplitude' : [.3, .0], # initial burst, then silent
        #     # 'amplitude' : [.37, .0], # postburst subthreshold oscillatory dynamic
        #     'amplitude' : [.5, .0], # chattering at ~5Hz
        #     # 'amplitude' : [.5, .0], # tonic
        #     # 'amplitude' : [.36, .0], # 

        #     # protocol as in figure 2C AvanziniCurtisPanzicaSpreafico1989
        #     # 'amplitude' : [.10, .0], # no result
        #     # 'amplitude' : [.55, .0], # initial burst, then silent
        #     # 'amplitude' : [.58, .0], # 
        #     # 'amplitude' : [.95, .0], # postburst subthreshold oscillatory dynamic
        #     # 'amplitude' : [1., .0], # chattering at ~1Hz
        #     # 'amplitude' : [1.1, .0], # tonic

        #     # 'amplitude' : [.25, .0], # default
        #     'start' : [200., 850.], # long duration
        #     # 'start' : [200., 600.], # short duration
        #     'stop' : 0.0
        # },
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
        'PhaseSpace' : False,
        'Rasterplot' : False,
        'ISI' : False,
        'ISI#' : False,
        'CrossCorrelation' : False,
        'FiringRate' : False,
        'LFP' : False,
    },

}
