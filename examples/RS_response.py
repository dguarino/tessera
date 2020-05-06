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
        # ACh effects on short-term post-synaptic plasticity (PSP)
        'ext' : {
            'n' : 1,
            # 'type': sim.SpikeSourceArray(spike_times=[100.]), # 0.1 Hz
            'type': sim.SpikeSourceArray(spike_times=[100., 125., 150., 175., 200., 225., 250., 275., 300., 325.]), # 40 Hz, TsodyksMarkram1997, 
            # 'type': sim.SpikeSourceArray(spike_times=[100., 133.3, 166.6, 199.9, 233.2, 266.5, 299.8, 333.1, 366.4, 399.7]), # 30 Hz, TsodyksMarkram1997
            # 'type': sim.SpikeSourceArray(spike_times=[100., 150., 200., 250., 300., 350., 400., 450., 500.,550.]), # 20 Hz, TsodyksMarkram1997
            # 'type': sim.SpikeSourceArray(spike_times=[100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.]), # 10 Hz, TsodyksMarkram1997
            'cellparams': None,
        },

        'cell' : {
            'n': 1, # units             
            'type': sim.EIF_cond_alpha_isfa_ista,
            'cellparams': {
                ##################################
                # ACh effects on input resistance

                'cm'         : 0.15,  # nF, tot membrane capacitance (Naud et al. 2008, https://www.neuroelectro.org/neuron/111/
                'tau_refrac' : 2.,    # ms, refractory period ()
                'delta_T'    : 0.8,   # mV, steepness of exponential approach to threshold (Naud et al. 2008)
                'v_thresh'   : -52.0, # mV, fixed spike threshold (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, 107)
                # 'v_thresh'   : RandomDistribution('normal', (-52.0, 2.0), rng=NumpyRNG(seed=85524)),

                # McCormickPrince1989: ACh increases input resistance, decreases leak conductance
                # ChauvetteCrochetVolgushevTimofeev2011: resting in cat natural sleep SW (-60mV UP, -80mV DOWN) 

                # # Control
                # 'tau_m'      : 17.0,  # ms, time constant of leak conductance (cm/gl, gl=8nS)
                # 'v_rest'     : -75.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -67.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'a'          : 1.0,   # nS, conductance of adaptation variable (Naud et al. 2008)
                # 'b'          : 0.05, # nA, increment to the adaptation variable (Naud et al. 2008)
                # 'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)

                # # ACh @resting -52mV (McCormickPrince1986 Fig. 3A holding -60mV)
                # 'tau_m'      : 30.0,  # ms, time constant of leak conductance (cm/gl, gl=5nS)
                # 'v_rest'     : -52.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                # 'v_reset'    : -67.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                # 'a'          : 1.0,   # nS, conductance of adaptation variable (Naud et al. 2008)
                # 'b'          : 0.005, # nA, increment to the adaptation variable (Naud et al. 2008)
                # 'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)

                # # ACh @resting -75mV (McCormickPrince1986 Fig. 3B holding -80mV)
                'tau_m'      : 20,  # ms, time constant of leak conductance (cm/gl, gl=7.5nS) PRE
                # 'tau_m'      : 25,  # ms, time constant of leak conductance (cm/gl, gl=6nS) POST1
                'v_rest'     : -75.0, # mV, resting potential E_leak (https://www.neuroelectro.org/neuron/111, 107)
                'v_reset'    : -55.0, # mV, reset after spike (Naud et al. 2008, https://www.neuroelectro.org/neuron/111, AHP Amplitude)
                'a'          : .1,   # nS, conductance of adaptation variable (Naud et al. 2008)
                'b'          : 0.1, # nA, increment to the adaptation variable (Naud et al. 2008)
                'tau_w'      : 88.0,  # ms, time constant of adaptation variable (Naud et al. 2008)

                ##################################
                # Cell level time constants of short-term post-synaptic plasticity (PSP)

                'tau_syn_E'  : 4.,  # TsodyksMarkram1997 Control and ACh  # ms, time constant of excitatory synaptic short-term plasticity

                'tau_syn_I'  : 2.0,  # Beierlein et al. 2003 Control # ms, time constant of excitatory synaptic short-term plasticity
            }
        },
    },

    'Projections' : {
        ##################################
        # No synaptic plasticity
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.StaticSynapse,
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     # 'delay' : 1., 
        # },

        ##################################
        # ACh effects on short-term post-synaptic plasticity (PSP)
        # # TC -> RS
        # 'ext_cell' : {
        #     'source' : 'ext',
        #     'target' : 'cell',
        #     'receptor_type' : 'excitatory',
        #     'synapse_type' : sim.TsodyksMarkramSynapse(U=0.3, tau_rec=30.0, tau_facil=20.0),
        #     'connector' : sim.AllToAllConnector(),
        #     'weight' : .1, # µS
        #     'delay' : 0.01, # ms
        # },

        # # RS -> RS
        'ext_cell' : {
            'source' : 'ext',
            'target' : 'cell',
            'receptor_type' : 'excitatory',
            # 'synapse_type' : sim.TsodyksMarkramSynapse(U=0.4, tau_rec=400.0, tau_facil=0.0), # Tsodyks and Markram 1997 Control
            'synapse_type' : sim.TsodyksMarkramSynapse(U=0.2, tau_rec=1900.0, tau_facil=0.0), # Tsodyks and Markram 1997 ACh
            'connector' : sim.AllToAllConnector(),
            'weight' : .0015, # µS
            'delay' : 0.01, # ms
        },

        # # FS -> RS
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
        # 'cell' : {
        #     ##################################
        #     # ACh effects on input resistance

        #     'source' : sim.StepCurrentSource,
        #     'amplitude' : [.15, .0], # depolarising
        #     'amplitude' : [-.15, .0], # hyperpolarising
        #     'start' : [200., 320.], # 
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
