"""
Copyright (c) 2016, Domenico GUARINO
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL GUARINO BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import csv
import shutil
from functools import reduce
import sys, getopt
import itertools as it
import numpy as np

from pyNN.utility import get_simulator
from pyNN.utility import init_logging
from pyNN.utility import normalized_filename
from pyNN.utility import Timer
from pyNN.space import Grid2D

import helpers as h
import analysis as a




# ADDITIONAL FUNCTIONS ---------------------------------------------------------

def replace(dic, keys,value):
    getValue(dic,keys[:-1])[keys[-1]]=value

def getValue(dic, keys):
    return reduce(lambda d, k: d[k], keys, dic)


# ------------------------------------------------------------------------------
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

sim, opts = get_simulator(
        ("--analysis", "Perform analysis only", {"type":bool}),
        ("--remove", "Remove data files (after analysis)", {"type":bool}),
        ("--folder", "Folder to save the data in (created if it does not exists)", {"dest":"data_folder", "required":True}),
        ("--params", "Parameter filename", {"dest":"param_file", "required":True}),
        ("--search", "Parameter search filename", {"dest":"search_file"}),
        ("--map",    "Produce a map of 2D parameter search", {"type":bool}),
        ("--debug", "Print debugging information")
    )

if opts.debug:
    init_logging(None, debug=True)

if opts.analysis:
    print("\nRunning analysis and plotting only ...")

if opts.remove:
    print("\nRemoving data files after analysis ...")

if opts.data_folder:
    print("\nData will be saved in:", opts.data_folder)
else:
    print("\nERROR: you must specify a folder name for the data files!")
    sys.exit(2)

params = {}
if opts.param_file != '':
    print("\nUsing parameter file:", opts.param_file)
    with open(opts.param_file, 'r') as pfile:
        pstring = pfile.read()
        params = eval(pstring)
        pfile.close()
else:
    print("\nERROR: you must specify a parameter file!")
    sys.exit(2)

search = {}
if opts.search_file:
    print("\nExecuting parameter search using file:", opts.search_file)
    with open(opts.search_file, 'r') as sfile:
        sstring = sfile.read()
        search = eval(sstring)




combinations = [{'default':''}] # init
if search:
    print(search)
    # create parameter combinations
    testParams = sorted(search) # give an order to dict (by default unordered)
    if len(testParams)>1:
        # create an array of dictionaries:
        # each dict being the joining of one of the testKey and a value testVal
        # each testVal is produced by internal product of all array in testParams
        combinations = [dict(list(zip(testParams, testVal))) for testVal in it.product(*(search[testKey] for testKey in testParams))]
    else:
        for testKey in testParams:
            print("testKey",testKey)

            combinations = []
            for testVal in search[testKey]:
                combinations.append({testKey:testVal})
            print(combinations)


# run combinations
info = []
totc = len(combinations)
for i,comb in enumerate(combinations):
    print("\n\nparam combination:",i,"/",totc)
    print("current set:",comb)

    # replacement
    for ckey,val in comb.items():
        keys = ckey.split('.') # get list from dotted string
        replace(params,keys,val)

    # save parameters in the data_folder
    if not os.path.exists(opts.data_folder):
        os.makedirs(opts.data_folder)

    param_file_name = opts.data_folder+'/'+str(comb)+'_'+opts.param_file
    with open(param_file_name, 'w') as pfile:
        pfile.write( str(params) )
        pfile.close()

    if not opts.analysis:
        Populations = h.build_network(sim, params)
        h.record_data(params, Populations)
        h.perform_injections(params, Populations)
        h.run_simulation(sim, params, Populations)
        h.save_data(Populations, opts.data_folder, str(comb))
        sim.end()

    scores = a.analyse(params, opts.data_folder, str(comb), opts.remove)

    # map storage of scores
    if search and opts.map:
        if i == 0:
            with open(opts.data_folder+'/map.csv', "w", newline="") as csvfile:
                fh = csv.writer(csvfile)
                if len(testParams)>1:
                    fh.writerow( ['#'+testParams[1]+':['+",".join(map(str, search[testParams[1]]))+"]" ] )
                    fh.writerow( ['#'+testParams[0]+':['+",".join(map(str, search[testParams[0]]))+"]" ] )
                else:
                    fh.writerow( ['#'+testParams[0]+':['+",".join(map(str, search[testParams[0]]))+"]" ] )


        info.append(scores) # current score
        
        if len(testParams)>1:
            if (i+1)%len(search[testParams[1]]) == 0:
                with open(opts.data_folder+'/map.csv', 'a') as csvfile:
                    fh = csv.writer(csvfile)
                    fh.writerow( info )
                    info = []
