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

import numpy as np

from pyNN.utility import Timer
from pyNN.space import Grid2D

from functools import partial # to run partial dt callback functions


# TODO: callback function to modify parameters while running the simulation
def modify_populations(t, params, populations):
    # populations['re'].set(tau_m=params['Populations']['cell']['cellparams']['tau_m'])
    if t>5000.0: # ms
        print(t)
        # spiketrains = populations['re'].get_data().segments[0]
        # get all tau_m: populations['re'][:].tau_m # because some of them will be different
        # for each cell:
        #   if fired:
        #       Ih = Ih + 'callback_params''increment'
        #       populations['re'][start:end].tau_m = params['tau_m'] + Ih
        #   else:
        #       exponential decay of Ih
        if populations['re'][0].tau_m > 12.:
            populations['re'].set(tau_m=(populations['re'][0].tau_m)-0.5)
            print( populations['re'][0].tau_m )
    return t + 50. #params['push_interval']



def build_network(sim, params):
    print("Setting up the Network ...")
    timer = Timer()
    timer.reset()

    sim.setup( timestep=params['dt'] )

    populations = {}
    for popKey,popVal in params['Populations'].items():

        number = popVal['n']
        if isinstance(popVal['n'],dict):
            number = int(params['Populations'][popVal['n']['ref']]['n'] * popVal['n']['ratio'])

        if 'structure' in popVal:
            populations[popKey] = sim.Population( number, popVal['type'], cellparams=popVal['cellparams'], structure=popVal['structure'])

            # printout network stats
            # populations[popKey].calculate_size(number)
            positions = popVal['structure'].generate_positions(number)
            #print(popKey, "shape:", positions.shape, "max position:", np.max(positions))
            # for cell in range(len(positions[0])):
            #     print(positions[0][cell], positions[1][cell])

        else:
            populations[popKey] = sim.Population( number, popVal['type'], cellparams=popVal['cellparams'])

        populations[popKey].initialize()

    projections = {}
    for projKey,projVal in params['Projections'].items():
        projections[projKey] = sim.Projection(
            populations[ projVal['source'] ],
            populations[ projVal['target'] ],
            connector = projVal['connector'],
            synapse_type = projVal['synapse_type'],
            receptor_type = projVal['receptor_type'],
            space = projVal['space'],
        )
        if 'weight' in projVal:
            projections[projKey].set(weight=projVal['weight'])
        if 'delay' in projVal:
            projections[projKey].set(delay=projVal['delay'])

        # # printout connectivity stats
        # if not "ext" in projKey:
        #     print(projKey, "- projections:", projections[projKey].size())
        #     print(projKey, "- conns per neuron:",projections[projKey].size()/params['Populations'][projVal['target']]['n'])
        #     # projList = projections[projKey].get(["weight", "delay"], format="list")
        #     # print(projKey, "- conn per neuron (border):", projList[:20] ) # first 20 units
        #     # print(projKey, "- conn per neuron (central):", projList[int(len(projList)/2)-20:int(len(projList)/2)+20] ) # mid 40 units
        #     # mean_dist = 0
        #     # for conn in projList:
        #     #     a = np.array([conn[0]%64, conn[0]/64])
        #     #     b = np.array([conn[1]%64, conn[1]/64])
        #     #     mean_dist += np.linalg.norm(a-b)
        #     # mean_dist /= len(projList)
        #     # # mean_dist 
        #     # print(projKey, "- mean conn distance", mean_dist)

    for modKey,modVal in params['Modifiers'].items():
        if type(modVal['cells']['start']) == float:
            start = int(modVal['cells']['start'] * populations[modKey].local_size)
        else:
            start = modVal['cells']['start']
        if type(modVal['cells']['end']) == float:
           end = int(modVal['cells']['end'] * populations[modKey].local_size)
        else:
            end = modVal['cells']['end']

        cells = populations[modKey].local_cells
        for key,value in modVal['properties'].items():
            populations[modKey][ populations[modKey].id_to_index(list(cells[ start:end ])) ].set(**{key:value})

    simCPUtime = timer.elapsedTime()
    print("... the simulation took %s ms to setup." % str(simCPUtime))

    return populations


def run_simulation(sim, params, populations):
    print("Running Network ...")
    timer = Timer()
    timer.reset()
    sim.run(params['run_time'])
    # modpop = partial(modify_populations, populations=populations)
    # sim.run(params['run_time'], callbacks=[modpop])
    simCPUtime = timer.elapsedTime()
    print("... the simulation took %s ms to run." % str(simCPUtime))


def perform_injections(params, populations):
    for modKey,modVal in params['Injections'].items():
        # if isinstance(modVal['spike_times'], (list)):
        #     source = modVal['source'](spike_times=modVal['spike_times'])
        # elif
        if isinstance(modVal['start'], (list)):
            source = modVal['source'](times=modVal['start'], amplitudes=modVal['amplitude'])
        else:
            source = modVal['source'](amplitude=modVal['amplitude'], start=modVal['start'], stop=modVal['stop'])
        populations[modKey].inject( source )


def record_data(params, populations):
    ## Record positions
    # populations = {}
    # for popKey,popVal in params['Populations'].items():
    #     print("population positions:", populations[popKey].positions)
    ## Record data
    for recPop, recVal in params['Recorders'].items():
        for elKey,elVal in recVal.items():
            #populations[recPop].record( None )
            if elVal == 'all':
                populations[recPop].record( elKey )
            elif 'MUA' in elVal:
                ids = populations[recPop].local_cells
                edge = int(np.sqrt(len(ids)))
                ids.shape = (edge, edge)
                assert elVal['x'] < edge, "MUA Recorder definition: x is bigger than the Grid2D edge"
                assert elVal['x']+elVal['size']+1 < edge, "MUA Recorder definition: x+size is bigger than the Grid2D edge"
                # print(elVal['x'],elVal['x']+elVal['size']+1)
                mualist = ids[ elVal['x']:elVal['x']+elVal['size']+1, elVal['y']:elVal['y']+elVal['size']+1 ].flatten()
                # print(mualist)
                populations[recPop][mualist.astype(int)].record( elKey )
            elif 'random' in elVal:
                populations[recPop].sample(elVal['sample']).record( elKey )
            else:
                populations[recPop][elVal['start']:elVal['end']].record( elKey )


def save_data(populations, folder, addon=''):
    print("Saving Data ...")
    timer = Timer()
    timer.reset()

    for key,p in populations.items():
        if key != 'ext':
            data = p.get_data()
            p.write_data(folder+'/'+key+addon+'.pkl', annotations={'script_name': __file__})

    simCPUtime = timer.elapsedTime()
    print("... the simulation took %s ms to save data." % str(simCPUtime))


