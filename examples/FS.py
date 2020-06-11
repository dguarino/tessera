# ------------------------------------------------------------------------------
# Usage:
# Examples

# start the docker image:
# $ docker run -v `pwd`:`pwd` -w `pwd` -i -t tessera /bin/bash

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

    'run_time': 1000., # ms
    'dt': 0.01, # ms

    'Populations' : {

        ##################################
        # # 1 neuron population to simulate ACh effects on short-term post-synaptic plasticity (PSP)
        # 'ext' : {
        #     'n' : 1,
        #     # 'type': sim.SpikeSourceArray(spike_times=[100.]), # 0.1 Hz
        #     'type': sim.SpikeSourceArray(spike_times=[100., 125., 150., 175., 200., 225., 250., 275.]), # 40 Hz, TsodyksMarkram1997, 
        #     # 'type': sim.SpikeSourceArray(spike_times=[100., 133.3, 166.6, 199.9, 233.2, 266.5, 299.8, 333.1, 366.4, 399.7]), # 30 Hz, TsodyksMarkram1997
        #     # 'type': sim.SpikeSourceArray(spike_times=[100., 150., 200., 250., 300., 350., 400., 450., 500.,550.]), # 20 Hz, TsodyksMarkram1997, Levy et al. 2008
        #     # 'type': sim.SpikeSourceArray(spike_times=[100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.]), # 10 Hz, TsodyksMarkram1997
        #     'cellparams': None,
        # },

        'cell' : {
            'n': 1, # units
            'type': sim.EIF_cond_exp_isfa_ista, # exponentially-decaying post-synaptic conductance
            # 'type': sim.EIF_cond_alpha_isfa_ista, # alpha-function-shaped post-synaptic conductance
            'cellparams': {
                'cm'         : 0.059,  # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/106)
                'tau_refrac' : 2.,    # ms, refractory period ()
                'delta_T'    : 3.,    # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'v_thresh'   : -40.0, # mV, fixed spike threshold (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107, 106)
                # 'v_thresh'   : RandomDistribution('normal', (-40.0, 2.0), rng=NumpyRNG(seed=85524)),

                ##################################
                # ACh effects on input resistance

                # # Control (McCormickPrince1986 Fig. 11A)
                # 'tau_m'      : 5.0,  # ms, time constant of leak conductance (cm/gl, gl=11.8nS) s=F/S 10-9/10-9
                # 'v_rest'     : -56.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -54.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'a'          : 1.8,   # nS, conductance of adaptation variable (Naud et al. 2008)
                # 'b'          : 0.061, # nA, increment to the adaptation variable (Naud et al. 2008)
                # 'tau_w'      : 16.0,  # ms, time constant of adaptation variable (Naud et al. 2008)

                # ACh (McCormickPrince1986 Fig. 11A)
                'tau_m'      : 5.0,  # ms, time constant of leak conductance (cm/gl, gl=11.8nS) s=F/S 10-9/10-9
                'v_rest'     : -41.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : -45.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : 0.8,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : 0.081, # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 28.0,  # ms, time constant of adaptation variable (Naud et al. 2008)

                # # ACh (McCormickPrince1986 Fig. 3B)
                # 'tau_m'      : 20,  # ms, time constant of leak conductance (cm/gl, gl=7.5nS) PRE
                # 'tau_m'      : 25,  # ms, time constant of leak conductance (cm/gl, gl=6nS) POST1
                # 'v_rest'     : -75.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -55.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'a'          : .1,   # nS, conductance of adaptation variable (Naud et al. 2008)
                # 'b'          : 0.1, # nA, increment to the adaptation variable (Naud et al. 2008)
                # 'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)


                # # Control (Levy et al. 2008, BeierleinGibsonConnors2003)
                # 'tau_m'      : 8.8,  # ms, time constant of leak conductance (cm/gl, gl=11.8nS) s=F/S 10-9/10-9
                # 'v_rest'     : -64.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -54.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'a'          : 1.8,   # nS, conductance of adaptation variable (Naud et al. 2008)
                # 'b'          : 0.061, # nA, increment to the adaptation variable (Naud et al. 2008)
                # 'tau_w'      : 16.0,  # ms, time constant of adaptation variable (Naud et al. 2008)

                # # # ACh (Levy et al. 2008)
                # 'tau_m'      : 5.0,  # ms, time constant of leak conductance (cm/gl, gl=11.8nS) s=F/S 10-9/10-9
                # 'v_rest'     : -51.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -45.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'a'          : 0.8,   # nS, conductance of adaptation variable (Naud et al. 2008)
                # 'b'          : 0.081, # nA, increment to the adaptation variable (Naud et al. 2008)
                # 'tau_w'      : 28.0,  # ms, time constant of adaptation variable (Naud et al. 2008)
          
                ##################################
                # Cell level time constants of short-term post-synaptic plasticity (PSP)

                # 'tau_syn_E'  : 5.,    # Beierlein et al. 2003 control and ACh # ms, time constant of excitatory synaptic short-term plasticity
                'tau_syn_E'  : 25.,    # Levy et al. 2008 control and ACh # ms, time constant of excitatory synaptic short-term plasticity
                'tau_syn_I'  : 20.0,  # ms, time constant of inhibitory synaptic short-term plasticity
            },
        },
    },

    'Projections' : {
        ##################################
        # No synaptic plasticity
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.StaticSynapse(),
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     # 'delay' : 1., 
        # },

        ##################################
        # ACh effects on short-term post-synaptic plasticity (PSP)

        # # TC -> FS
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=0.3, tau_rec=30.0, tau_facil=20.0),
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     'delay' : 2., 
        # },

        # # RS -> FS
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'excitatory',
        #     # 'synapse_type' : sim.TsodyksMarkramSynapse(U=.4, tau_rec=100., tau_facil=0.0), # Beierlein et al. 2003 Control
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=.5, tau_rec=400., tau_facil=0.0), # Levy et al. 2008 Control
        #     # 'synapse_type' : sim.TsodyksMarkramSynapse(U=.4, tau_rec=50., tau_facil=0.0), # Levy et al. 2008 ACh
        #     'connector' : sim.AllToAllConnector(),
        #     # 'weight' : 0.005, # Beierlein et al. 2003 # µS
        #     'weight' : 0.001, # Levy et al. 2008 # µS
        #     'delay' : 0.01, # ms
        # },
        # # # FS -> FS
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'inhibitory',
        #     # 'synapse_type' : sim.TsodyksMarkramSynapse(U=.5, tau_rec=40., tau_facil=0.0), # Control
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=.2, tau_rec=40., tau_facil=0.0), # ACh
        #     'connector' : sim.AllToAllConnector(),
        #     # 'weight' : 0.005, # Beierlein et al. 2003 # µS
        #     'weight' : 0.005, # Levy et al. 2008 # µS
        #     'delay' : 0.01, # ms
        # },
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'inhibitory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=.25, tau_rec=20., tau_facil=0.0), # Beierlein et al. 2003 Control
        #     # 'synapse_type' : sim.TsodyksMarkramSynapse(U=.4, tau_rec=420., tau_facil=0.0), # GigoutJonesWierschkeDaviesWatsonDeisz2012
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : 0.03, # Beierlein et al. 2003 # µS
        #     'delay' : 0.01, # ms
        # },
    },

    'Injections' : {
        'cell' : {
            ##################################
            # ACh effects on input resistance

            'source' : sim.StepCurrentSource,
            'amplitude' : [.25, .0], # default
            'start' : [200., 320.], # 
            'stop' : 0.0,
        },

        # 'cell' : {
        #     'source' : sim.StepCurrentSource,
        #     'amplitude' : [.01, .0], # default
        #     'start' : [20., 1000.], # 
        #     'stop' : 0.0,
        # },
    },


    'Recorders' : {
        'cell' : {
            'spikes' :  'all',
            'v' : 'all',
            'w' : 'all',
        },
    },


    'Modifiers' :{
    },


    'Analysis' : {
        'Static' : True,
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
