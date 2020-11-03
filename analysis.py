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
import gc
import json
import pickle 
import sys
import resource
import collections
import random
import itertools
from functools import cmp_to_key
from itertools import zip_longest # analysis
import numpy as np

################################
import matplotlib
matplotlib.use('Agg') # to be used when DISPLAY is undefined
################################

import matplotlib.pyplot as plot
from neo.core import AnalogSignal # analysis
import quantities as pq
import scipy
import scipy.linalg
import scipy.signal as signal

from pyNN.utility.plotting import Figure, Panel # analysis





def analyse(params, folder, addon='', removeDataFile=False):
    print("\nAnalysing data...")

    # populations key-recorders match
    populations = {}
    for popKey,popVal in params['Populations'].items():
        # if popKey != 'ext': # if is not the drive, to be dropped
        #     if popKey in params['Recorders']:
        #         populations[popKey] = list(params['Recorders'][popKey].keys())
        # we do the analysis on what we recorded
        if popKey in params['Recorders']:
            populations[popKey] = list(params['Recorders'][popKey].keys())


    ###################################
    if 'Landscape' in params['Analysis'] and params['Analysis']['Landscape']:

        print('Landscape')

        vector = [
            'tau_syn_E', # ms
            'tau_syn_I', # ms
            'tau_refrac',# ms, refractory period
            'delta_T',   # mV, steepness of exponential approach to threshold
            'v_thresh',  # mV, fixed spike threshold
            'cm',        # nF, tot membrane capacitance
            'tau_m',     # ms, time constant of leak conductance (cm/gl, with gl=0.01)
            'v_rest',    # mV, resting potential
            'v_reset',   # mV, reset after spike
            'a',         # nS, spike-frequency adaptation
            'b',         # nA, increment to the adaptation variable
            'tau_w',     # ms, time constant of adaptation variable
        ]
        colors = [
            'gray', # 'tau_syn_E', # ms
            'gray', # 'tau_syn_I', # ms
            'gray', # 'tau_refrac',# ms, refractory period
            'gray', # 'delta_T',   # mV, steepness of exponential approach to threshold
            'gray', # 'v_thresh',  # mV, fixed spike threshold
            'gray', # 'cm',        # nF, tot membrane capacitance
            'deepskyblue', # 'tau_m',     # ms, time constant of leak conductance (cm/gl, with gl=0.01)
            'orangered', # 'v_rest',    # mV, resting potential
            'red', # 'v_reset',   # mV, reset after spike
            'green', # 'a',         # nS, spike-frequency adaptation
            'limegreen', # 'b',         # nA, increment to the adaptation variable
            'blue', # 'tau_w',     # ms, time constant of adaptation variable
        ]
        scale = [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [10., 1000.],# 'tau_m',
            [-90., -40.],# 'v_rest'
            [-90., -40.],# 'v_reset'
            [0., 28.],# 'a'
            [0., .4],# 'b'
            [10., 1000.],# 'tau_w'
        ]

        states = []
        for trial in params['trials']: # ['SW','Spindles','AI'] # ['SW','Delta','Spindles','AI','REM']
            states.append(trial['name'])
        state_range = range(len(states))
        # state_colors = [['cornflowerblue','blue','darkblue'],['limegreen','forestgreen','darkgreen']]
        state_colors = ['blueviolet','deeppink','violet']

        landscape = np.zeros( (len(vector), len(states)) )

        # populate landscape
        for key,_ in params['Populations'].items():
            if key=='stimulus' or key=='boot':
                continue
            print("\n",key)
            for state_idx,trial in enumerate(params['trials']):
                state = trial['name']
                # print("\n",state)
                for stableparam_idx,param in enumerate(vector[0:6]):
                    # print(stableparam_idx, param, params['Populations'][key]['cellparams'][param])
                    landscape[stableparam_idx][state_idx] = params['Populations'][key]['cellparams'][param]
                stableparam_idx += 1
                for param_idx,param in enumerate(vector[6:]):
                    # print(stableparam_idx+param_idx, param, trial['modify'][key][param])
                    landscape[stableparam_idx+param_idx][state_idx] = trial['modify'][key][param]
            print(landscape)

            # params
            fig, host = plot.subplots()
            fig.subplots_adjust(right=0.75)

            host.set_xlabel('states')
            host.set_ylim(0, 1)

            axs = []
            for param_idx,param_name in enumerate(vector[6:]):

                axs.append( host.twinx() ) # instantiate a second axes that shares the same x-axis

                axs[param_idx].spines["right"].set_position(("axes", 1.+((param_idx+.001)/10.)))
                # axs[param_idx].spines["right"].set_visible(True)
                axs[param_idx].set_ylim(scale[stableparam_idx+param_idx][0], scale[stableparam_idx+param_idx][1])
                # print(stableparam_idx,param_idx,stableparam_idx+param_idx)
                color = colors[stableparam_idx+param_idx]
                axs[param_idx].plot(state_range, landscape[stableparam_idx+param_idx], marker='o', markeredgecolor=color, color=color, label=param_name)
                axs[param_idx].tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            fig.savefig(folder+'/Landscape_'+key+addon+'.svg', transparent=True)
            fig.clf()
            plot.close()

            # Dynamical Landscape
            fig = plot.figure()
            for state_idx,state_name in enumerate(states):
                I = 0 # at rest
                par = dict(zip(vector,landscape[:,state_idx]))
                # print(params)
                xn1, xn2 = nullcline( v_nullcline, par, I, (-100,0), 100 )
                yn1, yn2 = nullcline( w_nullcline, par, I, (-100,0), 100 )
                plot.plot(xn1, xn2, '-', linewidth=8-((state_idx+1)*2), color=state_colors[state_idx], label="Vm nullcline "+state_name)
                plot.plot(yn1, np.array(yn2)/1000, '-', linewidth=8-((state_idx+1)*2), color=state_colors[state_idx], label="W nullcline "+state_name)
            plot.axis([-100,-30,-.4,.6])
            plot.xlabel("V (mV)") 
            plot.ylabel("w (nA)") 
            # plot.legend(loc="lower right")
            plot.title("Phase space") 
            fig.savefig(folder+'/PhaseLandscape_'+key+addon+'.svg', transparent=True)
            plot.close()
            fig.clf()


    ###################################
    # transient
    if not 'transient' in params['Analysis']:
       params['Analysis']['transient'] = 100 # ms, INIT in case of older param files


    ###################################
    # analyses dependent on the static values of parameters, not the running itself
    if 'Static' in params['Analysis'] and params['Analysis']['Static']:
        # Mallory
        xn1, xn2 = H( params, (-50,10), 100 )
        fig = plot.figure()
        plot.plot(xn1, xn2)
        # plot.plot(vm,w,linewidth=2, color="red" )
        # plot.plot(xn1, xn2, '--', color="black")
        # plot.plot(xI1, xI2, color="black")
        # plot.plot(yn1, np.array(yn2)/1000, color="blue")
        # plot.axis([-100,-30,-.4,.6])
        plot.xlabel("V (mV)") 
        plot.ylabel("H") 
        plot.ylim([-10.,1000000.0])
        plot.title("H space") 
        fig.savefig(folder+'/H_'+addon+'.svg', transparent=True)
        fig.clf()
        plot.close()


    ###################################
    # phase synchrony
    if 'Coherence' in params['Analysis'] and params['Analysis']['Coherence']:
        print("Coherence needs to be updated")
        # # print(params['Analysis']['Coherence']['Population1'])
        # neo1 = pickle.load( open(folder+'/'+params['Analysis']['Coherence']['Population1']+addon+'.pkl', "rb") )
        # data1 = neo1.segments[0]
        # fr1 = firingrate(timeslice_start, params['run_time'], data1.spiketrains, bin_size=10) # ms
        # # print(params['Analysis']['Coherence']['Population1'], fr1.mean())
        # neo2 = pickle.load( open(folder+'/'+params['Analysis']['Coherence']['Population2']+addon+'.pkl', "rb") )
        # data2 = neo2.segments[0]
        # fr2 = firingrate(timeslice_start, params['run_time'], data2.spiketrains, bin_size=10) # ms
        # # print(params['Analysis']['Coherence']['Population2'], fr2.mean())
        # print('Coherence')
        # phase_coherence(fr1, fr2, folder=folder, addon=addon)


    ###################################
    if 'Injections' in params['Analysis'] and params['Analysis']['Injections']:
        amplitude = np.array([0.]+params['Injections']['cell']['amplitude']+[0.])#[0.,-.25, 0.0, .25, 0.0, 0.]
        start = np.array([0.]+params['Injections']['cell']['start']+[params['run_time']])/params['dt']
        start_int = start.astype(int)
        current = np.array([])

        for i in range(1,len(amplitude)):
            if current.shape == (0,):
                current = np.ones( (start_int[i]-start_int[i-1]+1,1) )*amplitude[i-1]
            else:
                current = np.concatenate( (current, np.ones( (start_int[i]-start_int[i-1],1) )*amplitude[i-1]), 0)
        current = AnalogSignal(current, units='mA', sampling_rate=params['dt']*pq.Hz)
        current.channel_index = np.array([0])
        panels.append( Panel(current, ylabel = "Current injection (mA)",xlabel="Time (ms)", xticks=True, legend=None) )


    ###################################
    # iteration over populations and selective plotting based on params and available recorders
    for key,rec in populations.items():
        print("\n\nAnalysis for:",key)

        # assuming 2D structure to compute the edge N
        n = 0
        if isinstance(params['Populations'][key]['n'], dict):
            n = int(params['Populations'][params['Populations'][key]['n']['ref']]['n'] * params['Populations'][key]['n']['ratio'])
        else:
            n = int(params['Populations'][key]['n'])
        edge = int(np.sqrt(n))

        # trials
        for trial_id,trial in enumerate(params['trials']):
            print("\n"+trial['name'])

            for itrial in range(trial['count']):
                print("trial #",itrial)
                timeslice_start = params['run_time'] * trial_id + params['Analysis']['transient'] # to avoid initial transient
                timeslice_end   = params['run_time'] * (trial_id+itrial+1) 
                print("trial-based slicing window (ms):", timeslice_start, timeslice_end)

                # get data
                print("from file:",key+addon+'_'+trial['name']+str(itrial))
                neo = pickle.load( open(folder+'/'+key+addon+'_'+trial['name']+str(itrial)+'.pkl', "rb") )
                data = neo.segments[0]

                # getting and slicing data
                # continuous variables (Vm, Gsyn, W) are sliced just according to the dt
                if 'w' in rec:
                    w = data.filter(name = 'w')[0]#[timeslice_start:timeslice_end]
                if 'v' in rec:
                    vm = data.filter(name = 'v')[0]#[timeslice_start:timeslice_end]
                if 'gsyn_exc' in rec:
                    gexc = data.filter(name = 'gsyn_exc')[0]#[timeslice_start:timeslice_end]
                if 'gsyn_inh' in rec:
                    ginh = data.filter(name = 'gsyn_inh')[0]#[timeslice_start:timeslice_end]
                # discrete variables (spiketrains) are sliced according to their time signature
                if 'spikes' in rec:
                    # spiketrains = data.spiketrains[ (data.spiketrains[:]>=timeslice_start) & (data.spiketrains[:]<=timeslice_end) ]
                    spiketrains = []
                    for spiketrain in data.spiketrains:
                        spiketrains.append(spiketrain[ (spiketrain>=timeslice_start) & (spiketrain<=timeslice_end) ])


                panels = []
                # return list for param search
                scores = []
                scores.append(0)   # 0. Spike count
                scores.append(0.0) # 1. Inter-Spike Interval
                scores.append(0.0) # 2. Coefficient of Variation
                scores.append("")  # 3. additional label for adapting ISI
                scores.append(0.0) # 4. Firing rate
                scores.append(0.0) # 5. mean power in the Delta range (0.1-4 Hz) 
                scores.append(0.0) # 6. mean power in the Spindles range (7-14 Hz) 
                scores.append(0.0) # 7. Cross-Correlation


                ###################################
                if 'PhaseSpace' in params['Analysis'] and params['Analysis']['PhaseSpace']:
                    print('PhaseSpace')
                    # plot W
                    fig = plot.figure()
                    plot.plot(w,linewidth=2)
                    # plot.ylim([-100,-20.0]) # just to plot it nicely
                    # plot.ylim([-100,0.0])
                    fig.savefig(folder+'/w_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    fig.clf()
                    plot.close()
                    # plot Phase space
                    I = 0.0 # at rest
                    xn1, xn2 = nullcline( v_nullcline, params['Populations']['cell']['cellparams'], I, (-100,0), 100 )
                    if params['Injections'] and params['Injections']['cell']['amplitude']:
                        I = params['Injections']['cell']['amplitude'][0]
                    xI1, xI2 = nullcline( v_nullcline, params['Populations']['cell']['cellparams'], I, (-100,0), 100 )
                    yn1, yn2 = nullcline( w_nullcline, params['Populations']['cell']['cellparams'], I, (-100,0), 100 )

                    # plotting
                    fig = plot.figure()
                    plot.plot(vm,w,linewidth=2, color="red" )
                    plot.plot(xn1, xn2, '--', color="black")
                    plot.plot(xI1, xI2, color="black")
                    plot.plot(yn1, np.array(yn2)/1000, color="blue")
                    plot.axis([-100,-30,-.4,.6])
                    # define a grid and compute direction at each point 
                    qx = np.linspace( -100, -30, 10)
                    qy = np.linspace( -.4, .6, 10)
                    X1 , Y1  = np.meshgrid(qx, qy) # create grid
                    DX1 = X1.copy()
                    DY1 = Y1.copy()
                    for j in range(len(qy)):
                        for i in range(len(qx)):
                            x1, y1 = Euler( _v, _w, params['Populations']['cell']['cellparams'], params['Injections']['cell']['amplitude'][0], (qx[i],qy[j]), params['dt'], 2) # compute 1 step growth
                            # print(x1, y1)
                            DX1[j][i] = x1[1]-x1[0]
                            DY1[j][i] = y1[1]-y1[0]
                    # print(DX1, DY1)
                    plot.quiver(X1, Y1, DX1, DY1, color='red', pivot='mid')
                    # Dmagnitude = np.sqrt(DX1**2 + DY1**2)
                    # DX1 = DX1 / Dmagnitude 
                    # DX1 = DX1 / Dmagnitude 
                    # scaling_factor = 0.0001
                    # plot.quiver(X1, Y1, DX1*scaling_factor, DY1*scaling_factor, color='red', pivot='mid', units="xy", scale_units='xy')
                    # plot.quiver(X1, Y1, DX1, DY1, color='red', pivot='mid', units="xy", scale=.01)
                    # plot.streamplot(X1, Y1, DX1, DY1)
                    # plot.quiver(X1, Y1, DX1, DY1, color='red', pivot='mid', angles='xy', scale_units='xy', scale=1)
                    # M = (np.hypot(DX1, DY1))
                    # plot.rcParams['image.cmap'] = 'RdPu' # rose
                    # plt.rcParams['image.cmap'] = 'PuRd' # pink
                    # plt.rcParams['image.cmap'] = 'Purples' # violet
                    # plot.quiver(X1, Y1, DX1, DY1, M, pivot='mid')

                    plot.xlabel("V (mV)") 
                    plot.ylabel("w (nA)") 
                    plot.title("Phase space") 
                    fig.savefig(folder+'/phase_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clf()



                ###################################
                if 'Vm' in params['Analysis'] and params['Analysis']['Vm'] and key in params['Analysis']['Vm']:
                    print('Vm')
                    # print(vm)
                    fig = plot.figure()
                    plot.plot(vm,linewidth=2)
                    plot.ylim([-100,0.0]) # GENERIC
                    #################
                    # plot.xlim([9900,12500]) # E/IPSP single pulse (0.1 Hz)
                    #################
                    # plot.xlim([10000,60000]) # E/IPSP single pulse (0.1 Hz)
                    # plot.ylim([-66,-54]) # Control EPSP on RE
                    #################
                    # plot.ylim([-75.5,-71.5]) # Control EPSP on RS
                    # plot.ylim([-79.,-74.5]) # ACh EPSP on RS
                    # plot.ylim([-79.,-74.5]) # Control IPSP on RS
                    # plot.ylim([-79.,-74.5]) # ACh IPSP on RS
                    #################
                    # plot.ylim([-64.5,-60]) # Control EPSP on FS
                    # plot.ylim([-51.5,-47.]) # ACh EPSP on FS
                    # plot.ylim([-67.5,-63.5]) # Control IPSP on FS
                    # plot.ylim([-54.5,-50.5]) # ACh IPSP on FS
                    #################
                    # all Vms
                    fig.savefig(folder+'/vm_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clf()
                    #################
                    # Average Vms
                    fig = plot.figure()
                    plot.plot(np.mean(vm,axis=1),linewidth=2)
                    plot.ylim([-100,0.0]) # GENERIC
                    fig.savefig(folder+'/avg_vm_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clf()
                    #################
                    # Vm histogram
                    fig = plot.figure()
                    ylabel = key
                    n,bins,patches = plot.hist(np.mean(vm,1), bins=50, normed=True) # 50*dt = 5ms bin
                    fig.savefig(folder+'/Vm_histogram_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clear()



                ###################################
                if 'Rasterplot' in params['Analysis'] and key in params['Analysis']['Rasterplot']:
                    print('Rasterplot')

                    # cell selection
                    if key in params['Analysis']['Rasterplot']:
                        if params['Analysis']['Rasterplot'][key]['limits'] == 'all':
                            spikelist = select_spikelist( spiketrains=spiketrains, edge=edge )
                        else:
                            spikelist = select_spikelist( spiketrains=spiketrains, edge=edge, limits=params['Analysis']['Rasterplot'][key]['limits'] )

                    local_addon = addon
                    if params['Analysis']['Rasterplot']['interval']:
                        local_addon = local_addon +'_zoom'

                    fig = plot.figure()
                    for row,st in enumerate(spikelist):
                        train = st
                        if params['Analysis']['Rasterplot']['interval']:
                            train = train[ train>params['Analysis']['Rasterplot']['interval'][0] ]
                            train = train[ train<params['Analysis']['Rasterplot']['interval'][1] ]
                        plot.scatter( train, [row]*len(train), marker='o', edgecolors='none', s=0.2, c=params['Analysis']['Rasterplot'][key]['color'] )
                    fig.savefig(folder+'/spikes_'+key+local_addon+'_'+trial['name']+str(itrial)+params['Analysis']['Rasterplot']['type'], transparent=True, dpi=params['Analysis']['Rasterplot']['dpi'])
                    plot.close()
                    fig.clf()



                ###################################
                # we are iterating over the data for the key
                # we want to visualize data.spiketrain as a time set of 2D frames
                if 'Movie' in params['Analysis'] and params['Analysis']['Movie'] and 'spikes' in rec:
                    # print(key , params['Analysis']['Movie']['populations'])
                    if not key in params['Analysis']['Movie']['populations']:
                        continue # we only do the movie for the required population
                    print('Movie')

                    duration = int(params['run_time']/params['dt'])
                    frames = None

                    print('+ preparing frames ...')
                    # prapare frames for the biggest extent of space (among the population listed)
                    for k,v in params['Analysis']['Movie']['populations'].items():
                        # print(isinstance(k, dict))
                        # print(isinstance(v, dict))
                        # print(v['ratio'])
                        if v['ratio'] == 1: # biggest extent of network space
                            n = 0
                            if isinstance(params['Populations'][k]['n'], dict):
                                n = int(params['Populations'][params['Populations'][k]['n']['ref']]['n'] * params['Populations'][k]['n']['ratio'])
                            else:
                                n = int(params['Populations'][k]['n'])
                            refedge = int(np.sqrt(n))

                            shape = ( duration+1, refedge, refedge )
                            frames = np.zeros(shape)
                            break 

                    v = params['Analysis']['Movie']['populations'][key]
                    # print(isinstance(v, dict))

                    if v['plot']:
                        # iterate over spiketrains and use times to update frames
                        for idn,st in enumerate(spiketrains):
                            # use st entries as indexes to put ones
                            x = int((idn*v['ratio']) % refedge)
                            y = int((idn*v['ratio']) / refedge)
                            for t in st:
                                frames[int(t/params['dt'])][x][y] = 1.

                        if frames is not None:
                            print('+ saving frames ...')
                            # plot one figure per timestep merging the populations
                            for t,frame in enumerate(frames[params['Analysis']['Movie']['from']:params['Analysis']['Movie']['to']]):
                                fig = plot.figure()
                                for x in range(refedge):
                                    for y in range(refedge):
                                        # print(x,y, frame[y][x])
                                        if frame[x][y]:
                                            plot.scatter(x, y, marker='o', c=v['color'], edgecolors='none', s=10 )
                                        else:
                                            plot.scatter(x, y, marker='o', c='white', edgecolors='none', s=10 )
                                plot.title( "{0:6.1f}".format((t+params['Analysis']['Movie']['from'])*params['dt']) )
                                fig.savefig(folder+'/'+str(t)+'_'+trial['name']+str(itrial)+'.png')
                                # fig.savefig(folder+'/'+str(t)+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                                plot.close()
                                fig.clf()
                    # after just compose the video with ffmpeg from the numbered.png
                    # ffmpeg -r 10 -pattern_type glob -i '*.png' -pix_fmt yuv420p closed.mp4



                ###################################
                if 'Autocorrelation' in params['Analysis'] and params['Analysis']['Autocorrelation'] and 'spikes' in rec:
                    print('Autocorrelation')
                    # Spike Count
                    if hasattr(spiketrains[0], "__len__"):
                        scores[0] = len(spiketrains[0])
                    
                    print(key)
                    if key in params['Analysis']['Autocorrelation']['populations']:
                        # Autocorrelogram
                        ac = aCC(params['run_time'], spiketrains, bin_size=params['Analysis']['Autocorrelation']['bin_size'], auto=True)
                        for i,ag in enumerate(ac):
                            x = np.linspace(-1.*(len(ag)/2), len(ag)/2, len(ag))
                            fig = plot.figure()
                            plot.plot(x,ag,linewidth=2)
                            plot.ylabel('count')
                            plot.xlabel('Time (bin='+str(params['Analysis']['Autocorrelation']['bin_size']*params['dt'])+'ms)')
                            fig.savefig(folder+'/Autocorrelogram_'+key+addon+'_'+str(i)+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                            plot.close()
                            fig.clear()



                ###################################
                if 'ISI' in params['Analysis'] and key in params['Analysis']['ISI'] and 'spikes' in rec:
                    print('ISI')
                    # Spike Count
                    if hasattr(spiketrains[0], "__len__"):
                        scores[0] = len(spiketrains[0])

                    # cell selection
                    spikelist = select_spikelist( spiketrains=spiketrains, edge=edge, limits=params['Analysis']['ISI'][key]['limits'] )

                    # ISI
                    print("time:", params['run_time'], "bins:", params['run_time']/params['Analysis']['ISI'][key]['bin'] )
                    isitot = isi(spikelist, int(params['run_time']/params['Analysis']['ISI'][key]['bin']) )
                    # print("ISI", isitot, isitot.shape)
                    if isinstance(isitot, (np.ndarray)):
                        if len(isitot)>1:
                            scores[1] = 0.0 # 
                            scores[2] = 0.0 #

                            # ISI histogram
                            fig = plot.figure()
                            plot.semilogy(range(len(isitot)), isitot)
                            # plot.plot(range(len(isitot)), isitot) 
                            plot.title("mean:"+str(scores[1])+" CV:"+str(scores[2]))
                            plot.ylabel('count')
                            plot.xlabel('ISI (bin=50ms)')
                            fig.savefig(folder+'/ISI_histogram_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                            plot.close()
                            fig.clear()

                            if 'ISI#' in params['Analysis'] and params['Analysis']['ISI#']:
                                # if strictly increasing, then spiking is adapting 
                                # but check that there are no spikes beyond stimulation
                                # if spiketrains[0][-1] < params['Injections']['cell']['start'][-1] and all(x<y for x, y in zip(isitot, isitot[1:])):
                                if spiketrains[0][-1] < params['run_time']:
                                    if all(x<y for x, y in zip(isitot, isitot[1:])):
                                        scores[3] = 'adapting'
                                        # ISIs plotted against spike interval position
                                        fig = plot.figure()
                                        plot.plot(isitot,linewidth=2)
                                        plot.title("CV:"+str(scores[2])+" "+str(addon))
                                        # plot.xlim([0,10])
                                        plot.ylim([2,12.])
                                        plot.xlabel('Spike Interval #')
                                        plot.ylabel('ISI (ms)')
                                        fig.savefig(folder+'/ISI_interval_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                                        plot.close()
                                        fig.clf()



                ###################################
                if 'FiringRate' in params['Analysis'] and key in params['Analysis']['FiringRate'] and 'spikes' in rec:
                    print('FiringRate')

                    # spikelist = select_spikelist( spiketrains=spiketrains, edge=edge, limits=params['Analysis']['FiringRate'][key]['limits'] )

                    # for row,st in enumerate(spikelist):
                    #     train = st
                    #     if params['Analysis']['FiringRate']['interval']:
                    #         train = train[ train>params['Analysis']['FiringRate']['interval'][0] ]
                    #         train = train[ train<params['Analysis']['FiringRate']['interval'][1] ]

                    # firing rate
                    fr = firingrate(timeslice_start, timeslice_end, spiketrains, bin_size=10) # ms
                    # fr = firingrate(params, spikelist, bin_size=10) # ms
                    scores[4] = fr.mean()
                    scores[7] = CC(timeslice_end-timeslice_start, spiketrains, bin_size=params['Analysis']['FiringRate']['bin'])
                    cvtot = cv(spiketrains, int(params['run_time']/params['Analysis']['FiringRate']['bin']) )
                    fig = plot.figure()
                    plot.plot(fr,linewidth=0.5)
                    plot.title("mean rate: %.2f (\pm %.2f) sp/s - CV: %.2f - CC: %.3f" % (scores[4], fr.std(), cvtot, scores[7]) )
                    plot.ylim(params['Analysis']['FiringRate'][key]['firing'])
                    fig.savefig(folder+'/firingrate_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    # plot.xlim([3000.,4000.])
                    # fig.savefig(folder+'/zoom_firingrate_'+key+addon+'.svg')
                    plot.close()
                    fig.clf()
                    ###################
                    ## spectrum
                    fig = plot.figure()
                    Fs = 1 / params['dt']  # sampling frequency
                    sr = Fs**2 # sample rate
                    # Compute the power spectrum 'classical way', with 2sec temporal window and 1sec overlap
                    freq, P = signal.welch(fr, sr, window='hamming')
                    # plot different spectrum types:
                    sp = plot.semilogx(freq, P, color = 'r')
                    delta = sp[0].get_ydata()[1:12] # 0.1-4 Hz interval power values
                    spindle = sp[0].get_ydata()[18:35] # 7-14 Hz interval power values
                    scores[5] = delta.mean()
                    scores[6] = spindle.mean()
                    plot.xlabel('Frequency (Hz)')
                    plot.ylabel('Power spectrum (µV**2)')
                    fig.savefig(folder+'/FR_Spectrum_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clear()



                ###################################
                if 'CrossCorrelation' in params['Analysis'] and params['Analysis']['CrossCorrelation'] and key in params['Analysis']['CrossCorrelation'] and 'spikes' in rec:
                    print('CrossCorrelation')
                    # cross-correlation
                    scores[7] = aCC(params, spiketrains, bin_size=params['Analysis']['CrossCorrelation'][key]['bin_size'])
                    print(scores[7])
                    print("CC:", scores[7])



                ###################################
                if 'LFP' in params['Analysis'] and params['Analysis']['LFP'] and 'v' in rec and 'gsyn_exc' in rec:
                    print('LFP')
                    lfp = LFP(data)
                    fig = plot.figure()
                    plot.plot(lfp)
                    fig.savefig(folder+'/LFP_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clear()
                    # # spectrum
                    # fig = plot.figure()
                    # Fs = 1 / params['dt']  # sampling frequency
                    # plot.title("Spectrum")
                    # # plot.magnitude_spectrum(lfp, Fs=Fs, scale='dB', color='red')
                    # plot.psd(lfp, Fs=Fs)
                    # fig.savefig(folder+'/Spectrum_'+key+addon+'_'+str(trial_name)+str(trial)+'.png')
                    # fig.clear()



                ###################################
                if 'ConductanceBalance' in params['Analysis'] and params['Analysis']['ConductanceBalance'] and 'gsyn_exc' in rec and 'gsyn_inh' in rec:
                    if not key in params['Analysis']['ConductanceBalance']:
                        continue
                    if trial['name'] not in params['Analysis']['ConductanceBalance'][key]['trials']:
                        continue
                    print('Conductance Balance')

                    # Average conductances
                    avg_gexc = np.mean(gexc, axis=1)
                    avg_ginh = np.mean(ginh, axis=1)
                    # conductances
                    fig = plot.figure()
                    plot.plot(avg_gexc,linewidth=2, color='red')
                    plot.plot(avg_ginh,linewidth=2, color='blue')
                    plot.xlabel('Time (s)')
                    plot.ylabel('Conductance (µS)')
                    fig.savefig(folder+'/avg_conductance_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clf()
                    # Conductance balance — a measure of contrast between excitation and inhibition
                    avg_gbalance = avg_gexc / (avg_gexc+avg_ginh)
                    fig = plot.figure()
                    plot.plot(avg_gbalance,linewidth=2, color='black')
                    plot.title("Mean conductance ratio: %.2f" % (avg_gbalance.nanmean()) )
                    plot.xlabel('Time (s)')
                    plot.ylabel('Conductance ratio')
                    fig.savefig(folder+'/avg_conductanceratio_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clf()



                ###################################
                if 'Ensembles_Clustering' in params['Analysis'] and params['Analysis']['Ensembles_Clustering'] and 'spikes' in rec:
                    if not key in params['Analysis']['Ensembles_Clustering']:
                        continue
                    if trial['name'] not in params['Analysis']['Ensembles_Clustering'][key]['trials']:
                        continue
                    print('Ensembles_Clustering')

                    spiketrains = select_spikelist( spiketrains=spiketrains, edge=edge, limits=params['Analysis']['Ensembles_Clustering'][key]['limits'] )
                    print("number of spiketrains:", len(spiketrains))
                    # print("spiketrains interval:", spiketrains[0])

                    # consider additional populations
                    if 'add' in params['Analysis']['Ensembles_Clustering'][key]:
                        for added_pop in params['Analysis']['Ensembles_Clustering'][key]['add']:
                            # get data
                            print("add spiketrains from file:",added_pop+addon+'_'+trial['name']+str(itrial)+'.pkl')
                            add_neo = pickle.load( open(folder+'/'+added_pop+addon+'_'+trial['name']+str(itrial)+'.pkl', "rb") )
                            add_data = add_neo.segments[0]
                            for added_spiketrain in add_data.spiketrains:
                                spiketrains.append(added_spiketrain[ (added_spiketrain>=timeslice_start) & (added_spiketrain<=timeslice_end) ])
                            print("number of spiketrains with "+added_pop+":", len(spiketrains))

                    # Calcium imaging -like analysis
                    #
                    # this analysis is globally as in MillerAyzenshtatCarrilloYuste2014
                    # but contains optimisations suggested by Anton Filipchuk in ... :
                    #
                    # 1. Compute population instantaneous firing rate (bin)
                    #
                    # 2. Establish threshold for population events
                    #   2.1 by computing ISIs of the original spiketrains
                    #   2.2 reshuffle ISIs (100) times
                    #   2.3 compute the population instantaneous firing rate for each surrogate timebinned rasterplot
                    #   (if there are no temporal relationship between cells firing, the real data will not be statistically different from the surrogates)
                    #
                    # 3. Find population events in the trial (when are the notable moments?)
                    #   3.1 smoothed firing rate 
                    #   3.2 instantaneous threshold is the 99% of the surrogate population instantaneous firing rate
                    #   3.3 the intersections of smoothed fr and threshold give nominal start/end of population events
                    #   3.4 the extrema beyond threshold are the peaks and drought of the population events
                    #   3.5 the minima before and after a peak are taken as start and end times of the peak population event
                    #
                    # 4. Compare the composition of population events 
                    #   4.1 produce a cell id signature of each population event ensemble, in terms of string of cell ids firing during the event ensemble
                    #   4.2 perform clustering linkage by complete cross-correlation of ensembles
                    #   4.2 plot dendrogram of linkage
                    #   4.3 plot cross-correlation of ensembles ordered by dendrogram (the carpet)
                    #
                    # NOTEs: 
                    # - compare 
                    # - consider the possibility of a threshold based on per-cell firing rate
                    # - could there be more cell assemblies in a population event?

                    print("time:", params['run_time'], "bin size:",params['Analysis']['Ensembles_Clustering'][key]['bin'] , "bins:", (params['run_time']/(params['Analysis']['Ensembles_Clustering'][key]['bin']) ) )

                    print("1. Compute population instantaneous firing rate (bin)")
                    
                    fr = firingrate(timeslice_start, timeslice_end, spiketrains, bin_size=params['Analysis']['Ensembles_Clustering'][key]['bin']) # ms

                    print("2. Compute surrogates to establish population event threshold")

                    # 2.1 by computing ISIs of the original spiketrains
                    spiketrainsISI = []
                    for st in spiketrains:
                        spiketrainsISI.append( np.diff( st.magnitude ).astype(int) )

                    # 2.2 reshuffle ISIs (100) times
                    surrogate_fr = []
                    for isur in range(100):
                        # build surrogate rasterplot
                        surrogate_spiketrains = []
                        for isi in spiketrainsISI:
                            random.shuffle(isi) # in-place function
                            st = []
                            curt=timeslice_start
                            for it in isi:
                                curt += it
                                st.append(curt)
                            surrogate_spiketrains.append( st )

                        # 2.3 compute the population instantaneous firing rate for each surrogate timebinned rasterplot
                        surrogate_fr.append( firingrate(timeslice_start, timeslice_end, surrogate_spiketrains, bin_size=params['Analysis']['Ensembles_Clustering'][key]['bin']) )

                    print("3. Find population events in the trial")

                    #   3.1 smoothed version 
                    smooth_fr = smooth(fr,params['Analysis']['Ensembles_Clustering'][key]['smoothing_points']) # 10 points, each of 5 ms

                    #   3.2 instantaneous threshold is the 99% of the surrogate population instantaneous firing rate
                    event_threshold = np.percentile(np.array(surrogate_fr), 99, axis=0) # 99 percentile
                    smooth_threshold = smooth(event_threshold,params['Analysis']['Ensembles_Clustering'][key]['smoothing_points']) # 10 bins, each of 5 ms

                    #   3.3 the intersections of smoothed fr and threshold give nominal start/end of population events
                    smooth_fr_crossings = np.argwhere(np.diff(np.sign(smooth_fr - smooth_threshold))).flatten() # index of the crossing point
                    # print("crossing",smooth_fr_crossings)

                    #   3.4 the extrema beyond threshold are peaks and those below are drought of population events
                    # Maxima
                    smooth_fr_hi_extrema = []
                    for peak in signal.argrelextrema(smooth_fr, np.greater)[0]:
                        if smooth_fr[peak] < smooth_threshold[peak]:
                            continue # there are maxima below threshold
                        smooth_fr_hi_extrema.append(peak)
                    # Minima
                    smooth_fr_lo_extrema = []
                    for drought in signal.argrelextrema(smooth_fr, np.less)[0]:
                        if smooth_fr[drought] > smooth_threshold[drought]:
                            continue # there are minima beyond threshold
                        smooth_fr_lo_extrema.append(drought)
                    # print("minima",smooth_fr_lo_extrema)

                    #   3.5 the minima before a crossing and after (a peak and a crossing) are taken as start and end times of the peak population event = ensemble
                    ensembles = [] # [(0){'start':0, 'end':0}}
                    ensembles_count = 0
                    # find ensemble limits
                    for previous_cross, current_cross in zip(smooth_fr_crossings[::], smooth_fr_crossings[1::]):
                        # if there is a peak between the two, take it, otherwise move to the next pair
                        have_peak = False
                        for peak in smooth_fr_hi_extrema:
                            if peak > previous_cross and peak < current_cross:
                                have_peak = True
                                break
                        if not have_peak:
                            continue
                        # default start and end of an ensemble are the crossing
                        ensemble = {'start':previous_cross, 'end':current_cross, 'cids':[]} # init
                        # the minimum before a crossing is the real start of population event
                        for drought in smooth_fr_lo_extrema:
                            # search the drought just before previous_cross and after the end of the last ensemble
                            if drought<previous_cross:
                                if len(ensembles)>1 and drought>ensembles[ensembles_count-1]['end']: 
                                    ensemble['start'] = drought # keep updating the drought before the cross
                            else:
                                break # do not go beyond crossing
                        # print(ensemble)
                        ensembles.append(ensemble)
                        ensembles_count += 1

                    # plot everything so far, surrogates, original, smoothed, threshold, maxima and minima, ensemble
                    # fig = plot.figure()
                    fig, ax = plot.subplots()
                    for surfr in surrogate_fr:
                        plot.plot(surfr,linewidth=0.5,color='grey')
                    for ensemble in ensembles:
                        ax.axvspan(ensemble['start'], ensemble['end'], alpha=0.2, color='red')
                    ax.plot(fr,linewidth=0.5,color='black')
                    ax.plot(event_threshold,linewidth=0.5,color='magenta')
                    ax.plot(smooth_fr,linewidth=0.5,color='green')
                    x = np.arange(0,len(smooth_fr))
                    ax.plot(x[smooth_fr_crossings],  smooth_fr[smooth_fr_crossings], 'rx', markersize=2)
                    ax.plot(x[smooth_fr_hi_extrema], smooth_fr[smooth_fr_hi_extrema], 'ws', markersize=2)
                    ax.plot(x[smooth_fr_lo_extrema], smooth_fr[smooth_fr_lo_extrema], 'wo', markersize=2)
                    plot.ylim(params['Analysis']['Ensembles_Clustering'][key]['ylim'])
                    fig.savefig(folder+'/PopulationEvents_firingrate_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clf()

                    # Population event distribution
                    # avalanche plot (Kauffman)
                    # histogram of how many events have a certain size (number of cid per event)
                    # x: avg number of cid per event
                    # y: avg number of event
                    # We should get a lot of events with few cid, and few events with a lot of cid, but with possible unexpected peaks 

                    # Derrida plot
                    # ...

                    if len(ensembles)<2:
                        print("... not enough ensembles to perform clustering")
                        continue
                    # ----

                    # print("4. Signature of population events")

                    # 4.1 produce a cell id signature of each population event ensemble: string of cell ids firing during the event ensemble
                    # get the cell ids for each ensemble
                    ensembles_signatures = [] # N x M, N ensembles and M cids for each ensemble
                    for ensemble in ensembles:
                        signature = [0 for i in range(len(spiketrains))] # init
                        # start end of population event index along the smoothed fr are converted to ms in order to search cell ids being active in that window
                        tstart = (ensemble['start'] * params['Analysis']['Ensembles_Clustering'][key]['bin'] ) +timeslice_start
                        tend = (ensemble['end'] * params['Analysis']['Ensembles_Clustering'][key]['bin'] ) +timeslice_start
                        # print('\n',tstart,tend)
                        # print("start conv:", ensemble['start'] , params['Analysis']['Ensembles_Clustering'][key]['bin'])
                        # print("start conv:", ensemble['end'] , params['Analysis']['Ensembles_Clustering'][key]['bin'])
                        for idx,spiketrain in enumerate(spiketrains): # place the spiketrain cid in the signature at spiketrain index
                            # take the idx if there are spikes within ensemble start and end
                            present = np.where(np.logical_and(spiketrain>=tstart, spiketrain<=tend))[0] 
                            # print(idx,"present:",present)
                            if len(present) >= params['Analysis']['Ensembles_Clustering'][key]['ensemble_support']: # support number of times in the window
                                # print("id",spiketrain.annotations['source_id'], "present:",present, spiketrain)
                                signature[idx] = int(spiketrain.annotations['source_id']) # to store the actual cid and not just the index in spiketrain
                        ensembles_signatures.append( signature )
                        print("how many cid in this signature:",np.count_nonzero(signature))
                    ensembles_signatures = np.array(ensembles_signatures)
                    # if 'SW' in trial['name']:
                    #     for es in ensembles_signatures:
                    #         print(es[es>0])
                    #     print(ensembles_signatures, ensembles_signatures[ensembles_signatures>0])

                    print("Cross-Correlation of ensembles (Pearson's R)")
                    # Raw Pearson's correlation matrix over ensembles signatures (their patterns of cells)
                    PearsonR = np.corrcoef(ensembles_signatures).round(decimals=3)
                    # print(PearsonR)
                    fig = plot.figure()
                    plot.pcolor(PearsonR)
                    plot.colorbar()
                    fig.savefig(folder+'/Ensembles_CorrMatrix_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    # fig.savefig(folder+'/Matrix_'+key+addon+'.svg', transparent=True)
                    plot.close()
                    fig.clear()

                    # 4.2 perform clustering linkage by complete cross-correlation of ensemble signatures
                    # following: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
                    print("Clustering ...")
                    from scipy.cluster.hierarchy import dendrogram, linkage
                    from scipy.cluster.hierarchy import cophenet
                    from scipy.spatial.distance import pdist, squareform
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    import matplotlib as mpl
                    mpl.rcParams['lines.linewidth'] = 0.5
                    Z = linkage(ensembles_signatures, 'complete', metric='correlation') # as in matlab (Anton)
                    ccc, coph_dists = cophenet(Z, pdist(ensembles_signatures, metric='correlation')) # 
                    print("Cophenetic Correlation Coefficient:",ccc) # how much faithful is the dendrogram to the original data location (should be close to 1.0)
                    if ccc < 0.5:
                        print("... poor clustering (Cophenetic Coefficient < 0.5)")

                    # 4.2 plot dendrogram of linkage
                    fig = plot.figure()
                    dn = dendrogram(Z, orientation='right') # variable threshold asa in matlab
                    # dn = dendrogram(Z, orientation='right', color_threshold=1.) # fixed threshold
                    plot.yticks(fontsize=5)
                    plot.title("Cophenetic score: %.3f" % (ccc) )
                    fig.savefig(folder+'/Dendrogram_'+key+addon+'_'+trial['name']+str(itrial)+'.png', dpi=500, transparent=True)
                    # fig.savefig(folder+'/Dendrogram_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clear()
                    matplotlib.rcdefaults()

                    # 4.3 plot cross-correlation of ensembles ordered by dendrogram (the carpet)
                    # reorder the original ensembles to have the indexes as in the dendrogram
                    permutation = [int(x) for x in dn['leaves']]
                    clustered_signatures = ensembles_signatures[permutation]
                    # print(clustered_signatures)
                    # print(clustered_signatures.shape[1])
                    clustered_PearsonR = np.corrcoef(clustered_signatures).round(decimals=3)
                    fig = plot.figure()
                    plot.pcolor(clustered_PearsonR)
                    plot.colorbar()
                    locs = range(len(dn['leaves']))
                    xlocs,xlabels = plot.xticks()
                    ylocs,ylabels = plot.yticks()
                    plot.xticks(locs,dn['ivl'],fontsize=5)
                    plot.yticks(locs,dn['ivl'],fontsize=5)
                    fig.savefig(folder+'/ClusteredEnsembles_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    # # fig.savefig(folder+'/ClusteredEnsembles_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clear()

                    # Retrieving cluster colors for subsequent analyses
                    # retrieve the cluster groups from the colored branches in the dendrogram
                    # remove b from color_list
                    colorlist = [x for x in dn['color_list'] if x != 'b'] 
                    # the colorlist is made to represent edges between two points, we need the color of each leaf
                    # make new colorlist to match each leaf by removng blue and adding a same-color at the end of each group
                    currentcl = colorlist[0]
                    currentclidx = 0
                    clustermap_ensembles = []
                    colorlist_ensembles = []
                    clustermap = []
                    for cl in colorlist:
                        colorlist_ensembles.append(cl)
                        if cl != currentcl:
                            colorlist_ensembles.append(cl) # add one
                            # clustermap_ensembles.insert(clidx,currentcl)
                            clustermap_ensembles.append(currentcl)
                            clustermap.append(currentclidx)
                            currentcl = cl
                            currentclidx = currentclidx+1
                        clustermap_ensembles.append(cl)
                        clustermap.append(currentclidx)
                    colorlist_ensembles.append(cl)
                    clustermap_ensembles.append(cl)
                    clustermap.append(currentclidx)
                    # print("clustermap:",clustermap) # [0,0,1,1,1,1,2,2,2,...]
                    # print(clustermap_ensembles) # ['g', 'g', 'r', 'r', 'r', 'r', 'c', 'c', 'c', ...]
                    # print(colorlist_ensembles) # ['g', 'g', 'r', 'r', 'r', 'r', 'c', 'c', 'c', ...]

                    print("Clusters autocorrelation (cross-correlation within cluster)")
                    if ccc > 0.1:
                        # take minors of the clustered_PearsonR based on clustermap extremes
                        print("clustermap:",clustermap) # [0,0,1,1,1,1,2,2,2,...]
                        ensembles_group_indexes = []
                        cur_egroup = clustermap[0]
                        cur_egroupidx_start = 0
                        cur_egroupidx_end = 0
                        for egroupidx,egroup in enumerate(clustermap):
                            if egroup != cur_egroup:
                                ensembles_group_indexes.append([cur_egroupidx_start,cur_egroupidx_end])
                                cur_egroupidx_start = egroupidx
                                cur_egroup = egroup
                            cur_egroupidx_end = egroupidx
                        ensembles_group_indexes.append([cur_egroupidx_start,cur_egroupidx_end]) # last one is not recorded because is not different 
                        print(ensembles_group_indexes)
                        auto_cluster = {}
                        print(clustered_PearsonR.shape)
                        for clusterid,cgroup in enumerate(ensembles_group_indexes):
                            print(clusterid)
                            minor = np.delete(np.delete(clustered_PearsonR,cgroup[0],axis=0), cgroup[1], axis=1)
                            minor[minor==1.] = 0.
                            auto_cluster[clusterid] = minor
                        print( auto_cluster )


                    print("5. Identify core ensembles")
                    # Ensemble structure analysis

                    # Core ensembles
                    # histogram of the cids ensembles_signatures from the same cluster: core is those cid firing in 95% of ensembles
                    # TODO: remove possible confound (there might be cells very active in one ensemble and not in others), weight by cell participation to each ensemble
                    currentcl = colorlist_ensembles[0]
                    esignature = []
                    core_signatures = []
                    for ecids,ecolor in zip(clustered_signatures,colorlist_ensembles):
                        if ecolor != currentcl:
                            print(currentcl)
                            if len(esignature)>1:
                                fig = plot.figure()
                                h = plot.hist(esignature, clustered_signatures.shape[1], alpha=0.3, color=currentcl, linewidth=0.)
                                # print(h[0]) # histogram
                                print(np.argwhere(h[0] > (np.max(h[0])/100)*90).flatten() ) # 95%
                                core_signatures.append(np.argwhere(h[0] > (np.max(h[0])/100)*90).flatten()) # 90%
                                # fig.savefig(folder+'/Ensembles_signatures_'+key+addon+'_'+trial['name']+str(itrial)+str(currentcl)+".png", transparent=True, dpi=800)
                                fig.savefig(folder+'/Ensembles_signatures_'+key+addon+'_'+trial['name']+str(itrial)+str(currentcl)+".svg", transparent=True)
                                esignature = []
                                currentcl = ecolor
                        ecids = [i for i in ecids if i != 0]
                        esignature.extend(ecids)
                    plot.close()
                    fig.clf()

                    if 'print2Dmap' in params['Analysis']['Ensembles_Clustering'][key] and params['Analysis']['Ensembles_Clustering'][key]['print2Dmap']:
                        print("2D map of ensembles and core cells")
                        # plot a 2D map for each ensemble
                        # where 
                        # - empty circles are cells not active during the event
                        # - red circles are active and participating to the event
                        # read positions.txt into coords
                        coords = []
                        with open(folder+'/'+key+addon+'_'+trial['name']+str(itrial)+'_positions.txt', 'r') as posfile:
                            lines = posfile.readlines()
                            posfile.close()
                            for line in lines:
                                coords.append( line.split(' ') )
                        # print(coords)
                        for eid, (signature,core) in enumerate(zip(ensembles_signatures,core_signatures)):
                            # print(signature)
                            # scatter delle coordinate lette da file
                            # gli id nella signature vengono colorati
                            fig = plot.figure()
                            for coord in coords:
                                color = 'grey'
                                # print(coord[0], signature)
                                if int(coord[0]) in signature:
                                    color = 'red'
                                if int(coord[0]) in core:
                                    color = 'blue'
                                plot.scatter(coord[2], coord[3], marker='o', c=color, edgecolors='none', s=10 )
                            fig.savefig(folder+'/EnsembleMap'+str(eid)+'_'+trial['name']+str(itrial)+'.svg')
                            plot.close()
                            fig.clf()

                    # # Print raster in the same order of clustering
                    # # plot cid spiketrains reordered as in the branches
                    # fig = plot.figure()
                    # cidx = 0
                    # used_cid = []
                    # for esign,ecolor in zip(ensembles_signatures,colorlist_ensembles):
                    #     # it will have a row foreach cid spiketrain
                    #     for cid in esign:
                    #         for train in spiketrains:
                    #             if train.annotations['source_id']==cid and cid not in used_cid:
                    #                 # add train to plot
                    #                 plot.scatter( train, [cidx]*len(train), marker='o', edgecolors='none', s=0.2, c=ecolor )
                    #                 cidx = cidx+1
                    #                 used_cid.append(cid)
                    #                 break
                    # fig.savefig(folder+'/Ensembles_spikes_'+key+addon+'_'+trial['name']+str(itrial), transparent=True, dpi=800)
                    # plot.close()
                    # fig.clf()
                    # 
                    # fig = plot.figure()
                    # for row,train in enumerate(spiketrains):
                    #     plot.scatter( train, [row]*len(train), marker='o', edgecolors='none', s=0.2, c='k' )
                    # for coresign,ecolor in zip(core_signatures,colorlist_ensembles):
                    #     for cid in coresign:
                    #         for train in spiketrains:
                    #             if train.annotations['source_id']==cid:
                    #                 # add train to plot
                    #                 plot.scatter( train, [cid]*len(train), marker='o', edgecolors='none', s=0.2, c=ecolor )
                    #                 break
                    # fig.savefig(folder+'/Ensembles_spikes_'+key+addon+'_'+trial['name']+str(itrial), transparent=True, dpi=800)
                    # plot.close()
                    # fig.clf()

                    print("6. Cell assembly transitions")
                    # we want clustermap_ensembles ordered as range(ensembles_signatures)
                    # print(dn['leaves'])
                    # print(colorlist_ensembles)
                    zipped_lists = zip(dn['leaves'], colorlist_ensembles)
                    sorted_pairs = sorted(zipped_lists)
                    tuples = zip(*sorted_pairs)
                    ensembles_list, clustermap_ensembles = [ list(tuple) for tuple in tuples ]
                    # ensembles_list, clustermap_ensembles = (list(t) for t in zip(*sorted(zip(dn['leaves'], clustermap_ensembles))))
                    print(ensembles_list, len(ensembles_list))
                    print(clustermap_ensembles, len(clustermap_ensembles))
                    # plot sequence of colors
                    ensemble_groups_sequence = np.zeros( (len(np.unique(clustermap_ensembles)), max(ensembles_list)+1) ) 
                    # ensemble_groups_sequence = np.zeros( (len(np.unique(clustermap_ensembles)), len(ensembles_list)+1) ) 
                    for t,cl in zip(ensembles_list,clustermap_ensembles):
                        for cli,cll in enumerate(np.unique(clustermap_ensembles)):
                            if cl == cll:
                                ensemble_groups_sequence[cli][t] = cli+1
                                break
                    print(ensemble_groups_sequence)
                    #produce colormap with as many colors as there are unique values in df
                    from matplotlib.colors import ListedColormap
                    ensembles_colors = ['w']
                    ensembles_colors.extend(list(np.unique(clustermap_ensembles)))
                    cmap = ListedColormap(ensembles_colors)
                    fig = plot.figure()
                    plot.pcolor(ensemble_groups_sequence, cmap=cmap)
                    # fig.savefig(folder+'/AssembliesTransitions_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    fig.savefig(folder+'/AssembliesTransitions_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clear()

                    # Ensembles durations and intervals statistics
                    # to discriminate between permutations later on below.
                    ensembles_durations = [] # list for each assembly signature
                    ensembles_intervals = []
                    last_ensemble = None
                    fig, ax = plot.subplots()
                    for eid,ensemble in enumerate(ensembles):
                        color = 'grey' # very light grey
                        if eid in ensembles_list:
                            eidx = ensembles_list.index(eid)
                            color = clustermap_ensembles[eidx]
                        ax.axvspan(ensemble['start'], ensemble['end'], color=color)
                        ensembles_durations.append(ensemble['end']-ensemble['start'])
                        if last_ensemble: # only from the second time on
                            ensembles_intervals.append(ensemble['start']-last_ensemble['end'])
                        last_ensemble = ensemble
                    plot.ylim([0,10])
                    fig.savefig(folder+'/Assemblies_Ensambles_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    plot.close()
                    fig.clf()
                    # statistics
                    print("total mean duration", np.mean(ensembles_durations), np.std(ensembles_durations) )
                    print("total mean intervals", np.mean(ensembles_intervals), np.std(ensembles_intervals) )
                    # per assembly duration statistics
                    # ...
                    continue ##############################################################################################################################

                    # Statistics
                    # as in Abeles 1991, LeeWilson2002, Carrillo et al. 2008
                    print("Compute surrogates cell assemblies transitions (cell assemblies permutations)")
                    assemblies = list(np.unique(clustermap_ensembles))
                    print("Compute permutations of assemblies:",assemblies)
                    # permutations = N**R # with repetition
                    # where in turn R = N, N-1, ..., N-(len(N)-2) # -2 because we want at least a sequence group made of 2 assemblies
                    # loop for all possible permutations (shuffled transitions)
                    # R = N, permutations = 3**3
                    for r in range(2,len(assemblies)+1): # at least a sequence made of 2
                        print("sequences of length",r)
                        r_permutations = [p for p in itertools.product(assemblies, repeat=r)]
                        # print("permutations",len(r_permutations),r_permutations)
                        transition_threshold = 1/len(r_permutations)
                        count = [0]*len(r_permutations)
                        for iw,s in enumerate(window(clustermap_ensembles, n=r)):
                            # count occurrences of each permutation in r_permutations
                            if s in r_permutations:
                                # print("sequence",s, " with permutation index",r_permutations.index(s))
                                count[ r_permutations.index(s) ] += 1
                        # print(count)
                        print("the probability of getting a len(",r,") match by chance is",transition_threshold)
                        threshold_occurrences = transition_threshold*(iw+1)
                        print("which amounts to",threshold_occurrences,"occurrences")
                        # plot sequences occurrences
                        fig = plot.figure()
                        # plot one column per permutation (with count >= 1?)
                        # plot.hist(count, bins=range(len(r_permutations)+1))# one bin per permutation
                        plot.bar(range(len(r_permutations)), count, width=0.8, align='center')
                        # plot the significance threshold
                        plot.hlines(threshold_occurrences, xmin=0, xmax=len(r_permutations), colors='r', linestyles='dashed')
                        # labels
                        x = range(len(r_permutations))
                        labels = [''.join(a) for a in r_permutations]
                        plot.xticks(x, labels, rotation='vertical')
                        # plot.ylim(0,1)
                        fig.savefig(folder+'/Sequences_'+str(r)+'_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                        plot.close()
                        fig.clear()


                    # # clustermap_ensembles same order as dn['leaves'] and ensembles_signatures
                    # # clustermap_ensembles = ['m', 'r', 'c', ... N]
                    # # map color names onto numbers
                    # assembly_indexes = range(len(assemblies))
                    # Transitions = np.zeros( (len(np.unique(clustermap_ensembles)), len(np.unique(clustermap_ensembles))) )
                    # # get counts of transitions between ensemble pairs
                    # for s in window(clustermap_ensembles, n=2):
                    #     np.add.at(Transitions, (assemblies.index(s[0]), assemblies.index(s[1])), 1)
                    # print(Transitions)
                    # # probability of transition
                    # Probabilities = Transitions / Transitions.sum()
                    # print(Probabilities)
                    # fig = plot.figure()
                    # plot.pcolor(Probabilities)
                    # plot.colorbar()
                    # fig.savefig(folder+'/AssembliesTransitionsProb_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    # # # fig.savefig(folder+'/EnsemblesTransitionsProb_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    # plot.close()
                    # fig.clear()
                    # # # Condional probability of next ensemble given previous
                    # # Conditionals = Transitions / Transitions.sum(axis=-1, keepdims=True)
                    # # print(Conditionals)
                    # # fig = plot.figure()
                    # # plot.pcolor(Conditionals)
                    # # plot.colorbar()
                    # # fig.savefig(folder+'/EnsemblesTransitionsCond_'+key+addon+'_'+trial['name']+str(itrial)+'.png', transparent=True)
                    # # # # fig.savefig(folder+'/EnsemblesTransitionsProb_'+key+addon+'_'+trial['name']+str(itrial)+'.svg', transparent=True)
                    # # plot.close()
                    # # fig.clear()


                # Figure( *panels ).save(folder+'/'+key+addon+'_'+trial['name']+str(itrial)+".png")
                # Figure( *panels ).save(folder+'/'+key+addon+'_'+trial['name']+str(itrial)+".svg")

                # for systems with low memory :)
                if removeDataFile:
                    os.remove(folder+'/'+key+addon+'_'+trial['name']+str(itrial)+'.pkl')

                print("scores",key,":",scores)

    return scores # to fix: is returning only the last scores!




###############################################
# ADDITIONAL FUNCTIONS

from itertools import islice 
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def select_spikelist( spiketrains, edge=None, limits=None ):
    new_spiketrains = []
    for i,st in enumerate(spiketrains):

        # reject cells outside limits
        if limits:
            # use st entries as indexes to put ones
            x = int(i % edge)
            y = int(i / edge)
            # print(i, x,y)
            # if (x>10 and x<50) and (y>10 and y<54):
            if (x>limits[0][0] and x<limits[0][1]) and (y>limits[1][0] and y<limits[1][1]):
                # print('taken') 
                new_spiketrains.append( st )
        else:
            new_spiketrains.append( st )
        
    return new_spiketrains


from scipy.signal import hilbert, butter, filtfilt
"""
from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
Regrading low and high band pass limits:
For digital filters, Wn are in the same units as fs. 
By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency. 
(Wn is thus in half-cycles / sample.)
"""
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs # fs is 10 samples per ms (dt=0.1) therefore fs = 10 kHz; its Nyquist freq = fs/2
    low = lowcut / nyq # 5000 / 5000 = 1 (lowest period)
    high = highcut / nyq # 200 / 5000 = 0.04 (highest period)
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
# convolution window
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def phase_coherence(s1, s2, folder, addon, lowcut=5000., highcut=200., fs=10000, order=5):
    """
    from: 
    http://jinhyuncheong.com/jekyll/update/2017/12/10/Timeseries_synchrony_tutorial_and_simulations.html
    https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9

    Phase coherence is here the instantaneous phase synchrony measuring the phase similarities between two signals at each timepoint.
    The phase is to the angle of the signal when it is mapped between -pi to pi degrees. 
    When two signals line up in phase their angular difference becomes zero. 
    The angles are calculated through the hilbert transform of the signal. 
    Phase coherence can be quantified by subtracting the angular difference from 1.
    """
    s1 = butter_bandpass_filter(s1,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    s2 = butter_bandpass_filter(s2,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    # s1 = butter_bandpass_filter(s1+1.,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    # s2 = butter_bandpass_filter(s2+1.,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    # s1 = butter_bandpass_filter(s1[100:]/np.max(s1[100:]),lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    # s2 = butter_bandpass_filter(s2[100:]/np.max(s2[100:]),lowcut=lowcut,highcut=highcut,fs=fs,order=order)

    angle1 = np.angle(hilbert(s1),deg=False)
    angle2 = np.angle(hilbert(s2),deg=False)
    phase_coherence = 1-np.sin(np.abs(angle1-angle2)/2)
    N = len(angle1)

    # Plot results
    fig,ax = plot.subplots(3,1,figsize=(14,7),sharex=True)
    ax[0].plot(s1,color='r',label='fr1')
    ax[0].plot(s2,color='b',label='fr2')
    ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=2)
    ax[0].set(xlim=[0,N], title='Band-passed firing rate')
    ax[1].plot(angle1,color='r')
    ax[1].plot(angle2,color='b')
    ax[1].set(ylabel='Angle',title='Angle at each Timepoint',xlim=[0,N])
    ax[2].plot(phase_coherence)
    ax[2].set(ylim=[0,1.1],xlim=[0,N],title='Instantaneous Phase Coherence',xlabel='Time',ylabel='Coherence')
    plot.tight_layout()
    fig.savefig(folder+'/PhaseCoherence_'+addon+'.svg', transparent=True)
    plot.close()
    fig.clear()


def LFP(data):
    v = data.filter(name="v")[0]
    g = data.filter(name="gsyn_exc")[0]
    # g = data.filter(name="gsyn_inh")[0]
    # We produce the current for each cell for this time interval, with the Ohm law:
    # I = g(V-E), where E is the equilibrium for exc, which usually is 0.0 (we can change it)
    # (and we also have to consider inhibitory condictances)
    iex = g*(v) #AMPA
    # the LFP is the result of cells' currents, with the Coulomb law:
    avg_i_by_t = np.sum(i,axis=1)/i.shape[0] # no distance involved for the moment
    sigma = 0.1 # [0.1, 0.01] # Dobiszewski_et_al2012.pdf
    lfp = (1/(4*np.pi*sigma)) * avg_i_by_t
    return lfp



def adaptation_index(data):
    # from NaudMarcilleClopathGerstner2008
    k = 2
    st = data.spiketrains
    if st == []:
        return None
    # ISI
    isi = np.diff(st)
    running_sum = 0
    for i,interval in enumerate(isi):
        if i < k:
            continue
        print(i, interval)
        running_sum = running_sum + ( (isi[i]-isi[i-1]) / (isi[i]+isi[i-1]) )
    return running_sum / len(isi)-k-1



def CC( duration, spiketrains, bin_size=10, auto=False ):
    """
    Binned-time Cross-correlation
    """
    print("CC")
    if spiketrains == [] :
        return NaN
    # create bin edges based on number of times and bin size
    # binning absolute time, and counting the number of spike times in each bin
    bin_edges = np.arange( 0, duration, bin_size )
    # print("bin_edges", bin_edges.shape, bin_edges)
    counts = []
    for spike_times in spiketrains:
        counts.append( np.histogram( spike_times, bin_edges )[0] )# spike count of bin_size bins
    counts = np.array(counts)
    CC = np.corrcoef(counts)
    # print(CC.shape, CC)
    return np.nanmean(CC)


def aCC( duration, spiketrains, bin_size=10, auto=False ):
    """
    Binned-time Cross-correlation
    """
    print("use CC")
    # if spiketrains == [] :
    #     return NaN
    # # create bin edges based on number of times and bin size
    # # binning absolute time, and counting the number of spike times in each bin
    # bin_edges = np.arange( 0, duration, bin_size )
    # for i, spike_times_i in enumerate(spiketrains):
    #     # print(i)
    #     for j, spike_times_j in enumerate(spiketrains):
    #         if auto and i!=j:
    #             continue
    #         itrain = np.histogram( spike_times_i, bin_edges )[0] # spike count of bin_size bins
    #         jtrain = np.histogram( spike_times_j, bin_edges )[0]

    #         if auto:
    #             CC.append( np.correlate(itrain, jtrain, mode="full") )
    #         else:
    #             CC.append( np.corrcoef(itrain, jtrain)[0,1] )
    # if auto:
    #     # return np.sum(CC, axis=0)
    #     return CC
    # else:
    #     return np.nanmean(CC)



def firingrate( start, end, spiketrains, bin_size=10 ):
    """
    Population rate
    as in https://neuronaldynamics.epfl.ch/online/Ch7.S2.html
    """
    if spiketrains == [] :
        return NaN
    # create bin edges based on start and end of slices and bin size
    bin_edges = np.arange( start, end, bin_size )
    # print("bin_edges",bin_edges.shape)
    # binning total time, and counting the number of spike times in each bin
    hist = np.zeros( bin_edges.shape[0]-1 )
    for spike_times in spiketrains:
        hist = hist + np.histogram( spike_times, bin_edges )[0]
    return ((hist / len(spiketrains) ) / bin_size ) * 1000 # average over population; result in ms *1000 to have it in sp/s



def isi( spiketrains, bins ):
    """
    Mean Inter-Spike Intervals for all spiketrains
    """
    isih = np.zeros(bins)
    for st in spiketrains:
        # print("st diff (int)", np.diff( st.magnitude ).astype(int) )
        isih = isih + np.histogram( np.diff( st.magnitude ).astype(int), len(isih) )[0]
    return isih



def cv( spiketrains, bins ):
    """
    Coefficient of variation
    """
    ii = isi(spiketrains, bins)
    return np.std(ii) / np.mean(ii)



# # ----------------------------------
# # -----      Phase Space      ------
# # ----------------------------------
def _v(x,y,p,I):
    gL = p['cm'] / p['tau_m'] # tau_m = cm / gL
    return ( -gL*(x-p['v_rest']) + gL*p['delta_T']*np.exp((x-p['v_thresh'])/p['delta_T']) -y +I ) / p['cm'] # Brette Gerstner 2005

def v_nullcline(x,p,I):
    gL = p['cm'] / p['tau_m']
    return -gL*(x-p['v_rest']) + gL*p['delta_T']*np.exp((x-p['v_thresh'])/p['delta_T']) + I # Touboul Brette 2008

def _w(x,y,p):
    return ( p['a']*(x-p['v_rest']) ) / p['tau_w'] # Brette Gerstner 2005

def w_nullcline(x,p,I):
    return p['a']*(x-p['v_rest']) # Touboul Brette 2008

def nullcline(f, params, I, limits, steps): 
    fn = []
    c = np.linspace( limits[0], limits[1], steps ) # linearly spaced numbers
    for i in c:
        fn.append( f(i,params,I) )
    return c, fn

def Euler( f1, f2, params, Input, iv, dt, time): 
    x = np.zeros(time)
    y = np.zeros(time)
    # initial values: 
    x[0] = iv[0] 
    y[0] = iv[1]
    I = 0 # init
    # compute and fill lists
    i=1
    while i < time:
        # integrating
        x[i] = x[i-1] + ( f1(x[i-1],y[i-1],params,I) )*dt
        y[i] = y[i-1] + ( f2(x[i-1],y[i-1],params) )*dt
        # discontinuity
        if x[i] >= params['v_spike']:
            x[i-1] = params['v_spike']
            x[i] = params['v_spike']
            y[i] = y[i] + params['b']
        i = i+1 # iterate
    return x, y

# # ----------------------------------
# # ----------------------------------

def H(params, limits, steps): 
    p = params['Populations']['cell']['cellparams']
    # gL = p['cm'] / p['tau_m']
    cm = p['cm'] * pq.nF
    tau_m = p['tau_m']*pq.mS
    gL = cm / tau_m
    gc = gL/cm
    delta_T = p['delta_T']*pq.mV
    v_thresh = p['v_thresh']*pq.mV
    tau_w = p['tau_w']*pq.mS

    fn = []
    Vm = np.linspace( limits[0], limits[1], steps ) # linearly spaced numbers
    for v in Vm:
        v = v*pq.mV
        h = gc * (np.exp((v-v_thresh)/delta_T) -1.) - (1/tau_w)
        fn.append( h )
    return Vm, fn



