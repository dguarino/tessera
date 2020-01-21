{
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

    # apply constraints from the literature to reduce the dimensionality of the parameter space

    #'run_time' : [5000],
    #'Populations.py.n' : [1600],
    #'Modifiers.py.cells.end' : [0.12,0.13,0.14],
    #'Modifiers.py.properties.a' : [0.01, .02, 0.03],

    # 'Populations.py.cellparams.v_rest': np.arange(-80., -70., 2.),
    # 'Populations.py.cellparams.v_reset': np.arange(-75., -65., 2.),
    # # 'Populations.py.cellparams.a': np.arange(0.5, 5., .5),
    # # 'Populations.py.cellparams.b': np.arange(0.01, 0.1, .3),
    # 'Populations.inh.cellparams.v_rest': np.arange(-60., -50., 2.),
    # 'Populations.inh.cellparams.v_reset': np.arange(-80., -70., 2.),
    # 'Populations.inh.cellparams.a': np.arange(0.5, 2., .5),
    # 'Populations.inh.cellparams.b': np.arange(0.01, 0.1, .3),

    # 'Populations.inh.cellparams.v_rest': np.arange(-75., -60., 5.),
    # 'Populations.inh.cellparams.v_reset': np.arange(-75., -60., 5.),
    # 'Populations.inh.cellparams.a': np.arange(0.5, 5., .5),
    # 'Populations.inh.cellparams.b': np.arange(0.01, 0.1, .3),

    # 'Projections.py_py.weight': np.arange(.0001, .002, .0003), # nA
    # 'Projections.py_inh.weight': np.arange(.0001, .002, .0003), # nA
    # 'Projections.inh_inh.weight': np.arange(.01, .05, .01), # nA
    # 'Projections.inh_py.weight': np.arange(.01, .05, .01), # nA

    # 'Populations.inh.cellparams.a': np.arange(.0001, 0.002, 0.0005),

    # 'Populations.cell.cellparams.tau_m': np.arange(.0, 20., 1.), # 
    # 'Populations.cell.cellparams.v_rest': np.arange(-85., -55., 3), #
    # 'Populations.cell.cellparams.v_rest': np.arange(-65., -55., 0.5), # as in Cerina et al. 2015
    # 'Populations.cell.cellparams.v_rest': np.array([-58., -90.]), #np.arange(-70., -60., 2.),
    # 'Populations.cell.cellparams.v_rest': np.arange(-90., -58., 2.),
    # 'Populations.cell.cellparams.v_reset': np.arange(-90., -55., 7.),
    # 'Populations.cell.cellparams.v_thresh': np.arange(-90., -80., 2.), # mV: parameter search
    # 'Populations.cell.cellparams.v_thresh': np.arange(-60., -50., .5), # mV: parameter search
    # 'Populations.cell.cellparams.a': np.arange(.0, 36., 12.), # uS: Cerina et al. 2015
    # 'Populations.cell.cellparams.b': np.arange(.0, .00004, .000002), # nA

    # Search coarsely the whole vector
    # 'Populations.cell.cellparams.tau_m': np.arange(10., 20., 2.5), # 
    # 'Populations.cell.cellparams.v_rest': np.arange(-80., -50., 5.),
    # 'Populations.cell.cellparams.v_reset': np.arange(-70., -50., 5.),
    # 'Populations.cell.cellparams.a': np.arange(.001, 3., .5), # 
    # 'Populations.cell.cellparams.b': np.arange(.0001, .005, .001), # nA
    # 'Populations.py.cellparams.tau_w': np.arange(.01, 20., 2.), # 

    # 'Populations.tc.cellparams.a': np.arange(5., 35., 3.), # uS: Cerina et al. 2015
    # 'Populations.re.cellparams.a': np.arange(5., 35., 3.), # uS: Cerina et al. 2015

    # 'Populations.py.cellparams.a': np.arange(0.005, 4., .4), # uS: Cerina et al. 2015
    # 'Populations.inh.cellparams.a': np.arange(0.005, 4., .4), # uS: Cerina et al. 2015

    # 'Populations.py.cellparams.v_rest': np.arange(-90., -50., 5.), #
    # 'Populations.inh.cellparams.v_rest': np.arange(-90., -50., 5.), #
    # 'Populations.py.cellparams.tau_m': np.arange(.01, 20., 2.), # 
    # 'Populations.py.cellparams.b': np.arange(.02, .04, .002), # nA

    'Projections.py_tc.weight' : np.arange(.0005, .002, .0004), # nA
    'Projections.py_re.weight' : np.arange(.0005, .002, .0004), # nA
    'Projections.tc_py.weight' : np.arange(.0005, .002, .0004), # nA
    'Projections.tc_inh.weight': np.arange(.0005, .002, .0004), # nA


}
