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
import pickle 
from itertools import zip_longest # analysis
import numpy as np

################################
import matplotlib
matplotlib.use('Agg') # to be used when DISPLAY is undefined
################################

import matplotlib.pyplot as plot
from neo.core import AnalogSignal # analysis
import quantities as pq
import scipy.signal as signal

from pyNN.utility.plotting import Figure, Panel # analysis





def analyse(params, folder, addon='', removeDataFile=False):
    print("Analysing data...")

    # populations key-recorders match
    populations = {}
    for popKey,popVal in params['Populations'].items():
        if popKey != 'ext':
            if popKey in params['Recorders']:
                populations[popKey] = list(params['Recorders'][popKey].keys())


    ###################################
    # analyses dependent on the static values of parameters, not the running itself
    if params['Analysis']['Static']:
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
        # print(params['Analysis']['Coherence']['Population1'])
        neo1 = pickle.load( open(folder+'/'+params['Analysis']['Coherence']['Population1']+addon+'.pkl', "rb") )
        data1 = neo1.segments[0]
        fr1 = rate(params, data1.spiketrains, bin_size=10) # ms
        # print(params['Analysis']['Coherence']['Population1'], fr1.mean())
        neo2 = pickle.load( open(folder+'/'+params['Analysis']['Coherence']['Population2']+addon+'.pkl', "rb") )
        data2 = neo2.segments[0]
        fr2 = rate(params, data2.spiketrains, bin_size=10) # ms
        # print(params['Analysis']['Coherence']['Population2'], fr2.mean())
        print('Coherence')
        phase_coherence(fr1, fr2, folder=folder, addon=addon)


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
        print("Analysis for:",key)

        # assuming 2D structure to compute the edge N
        n = 0
        if isinstance(params['Populations'][key]['n'], dict):
            n = int(params['Populations'][params['Populations'][key]['n']['ref']]['n'] * params['Populations'][key]['n']['ratio'])
        else:
            n = int(params['Populations'][key]['n'])
        edge = int(np.sqrt(n))

        # get data
        neo = pickle.load( open(folder+'/'+key+addon+'.pkl', "rb") )
        data = neo.segments[0]

        panels = []

        # return list for param search
        scores = []
        scores.append(0)   # Spike count
        scores.append(0.0) # Inter-Spike Interval
        scores.append(0.0) # Coefficient of Variation
        scores.append("")  # additional label for adapting ISI
        scores.append(0.0) # Firing rate
        scores.append(0.0) # mean power in the Delta range (0.1-4 Hz) 
        scores.append(0.0) # mean power in the Spindles range (7-14 Hz) 
        scores.append(0.0) # Cross-Correlation


        ###################################
        if 'PhaseSpace' in params['Analysis'] and params['Analysis']['PhaseSpace'] and 'w' in rec and 'v' in rec:
            print('PhaseSpace')
            vm = data.filter(name = 'v')[0]
            w = data.filter(name = 'w')[0]
            # plot W
            fig = plot.figure()
            plot.plot(w,linewidth=2)
            # plot.ylim([-100,-20.0]) # just to plot it nicely
            # plot.ylim([-100,0.0])
            fig.savefig(folder+'/w_'+key+addon+'.svg', transparent=True)
            fig.clf()
            plot.close()
            # plot Phase space
            I = 0 # at rest
            xn1, xn2 = nullcline( v_nullcline, params, I, (-100,0), 100 )
            if params['Injections'] and params['Injections']['cell']['amplitude']:
                I = params['Injections']['cell']['amplitude'][0]
            else:
                I = 0.0
            xI1, xI2 = nullcline( v_nullcline, params, I, (-100,0), 100 )
            yn1, yn2 = nullcline( w_nullcline, params, I, (-100,0), 100 )
            fig = plot.figure()
            plot.plot(vm,w,linewidth=2, color="red" )
            plot.plot(xn1, xn2, '--', color="black")
            plot.plot(xI1, xI2, color="black")
            plot.plot(yn1, np.array(yn2)/1000, color="blue")
            plot.axis([-100,-30,-.4,.6])
            plot.xlabel("V (mV)") 
            plot.ylabel("w (nA)") 
            plot.title("Phase space") 
            fig.savefig(folder+'/phase_'+key+addon+'.svg', transparent=True)
            plot.close()
            fig.clf()


        ###################################
        if 'Vm' in params['Analysis'] and params['Analysis']['Vm'] and 'v' in rec:
            print('Vm')
            vm = data.filter(name = 'v')[0]
            # print(vm)
            # panels.append( Panel(vm, ylabel="Membrane potential (mV)", xlabel="Time (ms)", xticks=True, yticks=True, legend=None) )
            ###################################
            # workaround
            fig = plot.figure()
            plot.plot(vm,linewidth=2)
            plot.ylim([-100,0.0]) # GENERIC
            #################
            # plot.xlim([9900,12500]) # E/IPSP single pulse (0.1 Hz)
            #################
            plot.xlim([10000,60000]) # E/IPSP single pulse (0.1 Hz)
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
            # saving
            fig.savefig(folder+'/vm_'+key+addon+'.svg', transparent=True)
            # fig.savefig(folder+'/zoom_vm_'+key+addon+'.svg')
            plot.close()
            fig.clf()
            #################
            # Vm histogram
            fig = plot.figure()
            ylabel = key
            n,bins,patches = plot.hist(np.mean(vm,1), bins=50, normed=True) # 50*dt = 5ms bin
            fig.savefig(folder+'/Vm_histogram_'+key+addon+'.svg', transparent=True)
            plot.close()
            fig.clear()


        # if 'gsyn_exc' in rec:
        #     gsyn_exc = data.filter(name="gsyn_exc")[0]
        #     # panels.append( Panel(gsyn_exc,ylabel = "Exc Synaptic conductance (uS)",xlabel="Time (ms)", xticks=True,legend = None) )


        # if 'gsyn_inh' in rec:
        #     gsyn_inh = data.filter(name="gsyn_inh")[0]
        #     # panels.append( Panel(gsyn_inh,ylabel = "Inh Synaptic conductance (uS)",xlabel="Time (ms)", xticks=True,legend = None) )


        ###################################
        if 'Rasterplot' in params['Analysis'] and params['Analysis']['Rasterplot'] and 'spikes' in rec:
            print('Rasterplot')

            spikelist = data.spiketrains
            # cell selection
            if key in params['Analysis']['Rasterplot']:
                spikelist = select_spikelist( spiketrains=data.spiketrains, edge=edge, limits=params['Analysis']['Rasterplot'][key]['limits'] )

            local_addon = addon
            if params['Analysis']['Rasterplot']['interval']:
                local_addon = local_addon +'_zoom'

            fig = plot.figure()
            for row,st in enumerate(spikelist):

                train = st
                if params['Analysis']['Rasterplot']['interval']:
                    train = train[ train>params['Analysis']['Rasterplot']['interval'][0] ]
                    train = train[ train<params['Analysis']['Rasterplot']['interval'][1] ]

                plot.scatter( train, [row]*len(train), marker='o', edgecolors='none', s=0.2 )
            fig.savefig(folder+'/spikes_'+key+local_addon+params['Analysis']['Rasterplot']['type'], transparent=True, dpi=params['Analysis']['Rasterplot']['dpi'])
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
                for idn,st in enumerate(data.spiketrains):
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
                        fig.savefig(folder+'/'+str(t)+'.png')
                        # fig.savefig(folder+'/'+str(t)+'.svg', transparent=True)
                        plot.close()
                        fig.clf()
            # after just compose the video with ffmpeg from the numbered.png
            # ffmpeg -r 10 -pattern_type glob -i '*.png' -pix_fmt yuv420p closed.mp4



        ###################################
        if 'Autocorrelation' in params['Analysis'] and params['Analysis']['Autocorrelation'] and 'spikes' in rec:
            print('Autocorrelation')
            # Spike Count
            if hasattr(data.spiketrains[0], "__len__"):
                scores[0] = len(data.spiketrains[0])
            
            print(key)
            if key in params['Analysis']['Autocorrelation']['populations']:
                # Autocorrelogram
                ac = acc(params, data.spiketrains, bin_size=params['Analysis']['Autocorrelation']['bin_size'], auto=True)
                for i,ag in enumerate(ac):
                    x = np.linspace(-1.*(len(ag)/2), len(ag)/2, len(ag))
                    fig = plot.figure()
                    plot.plot(x,ag,linewidth=2)
                    plot.ylabel('count')
                    plot.xlabel('Time (bin='+str(params['Analysis']['Autocorrelation']['bin_size']*params['dt'])+'ms)')
                    fig.savefig(folder+'/Autocorrelogram_'+key+addon+'_'+str(i)+'.svg', transparent=True)
                    plot.close()
                    fig.clear()



        ###################################
        if 'ISI' in params['Analysis'] and params['Analysis']['ISI'] and key in params['Analysis']['ISI'] and 'spikes' in rec:
            print('ISI')
            # Spike Count
            if hasattr(data.spiketrains[0], "__len__"):
                scores[0] = len(data.spiketrains[0])

            # cell selection
            spikelist = select_spikelist( spiketrains=data.spiketrains, edge=edge, limits=params['Analysis']['ISI'][key]['limits'] )

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
                    fig.savefig(folder+'/ISI_histogram_'+key+addon+'.svg', transparent=True)
                    plot.close()
                    fig.clear()

                    if 'ISI#' in params['Analysis'] and params['Analysis']['ISI#']:
                        # if strictly increasing, then spiking is adapting 
                        # but check that there are no spikes beyond stimulation
                        # if data.spiketrains[0][-1] < params['Injections']['cell']['start'][-1] and all(x<y for x, y in zip(isitot, isitot[1:])):
                        if data.spiketrains[0][-1] < params['run_time']:
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
                                fig.savefig(folder+'/ISI_interval_'+key+addon+'.svg', transparent=True)
                                plot.close()
                                fig.clf()


        ###################################
        if 'FiringRate' in params['Analysis'] and params['Analysis']['FiringRate'] and 'spikes' in rec:
            print('FiringRate')

            # spikelist = select_spikelist( spiketrains=data.spiketrains, edge=edge, limits=params['Analysis']['ISI'][key]['limits'] )

            # firing rate
            fr = rate(params, data.spiketrains, bin_size=10) # ms
            # fr = rate(params, spikelist, bin_size=10) # ms
            scores[4] = fr.mean()
            fig = plot.figure()
            plot.plot(fr,linewidth=0.5)
            plot.title("mean firing rate:"+str(scores[4])+" spikes/s")
            plot.ylim(params['Analysis']['FiringRate'][key]['firing'])
            fig.savefig(folder+'/firingrate_'+key+addon+'.svg', transparent=True)
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
            # freq, P = signal.welch(fr, sr, window='hamming',nperseg=4000, noverlqp=nperseg*0.75 )
            # plot different spectrum types:
            sp = plot.semilogx(freq, P, color = 'r')
            delta = sp[0].get_ydata()[1:12] # 0.1-4 Hz interval power values
            spindle = sp[0].get_ydata()[18:35] # 7-14 Hz interval power values
            scores[5] = delta.mean()
            scores[6] = spindle.mean()
            plot.xlabel('Frequency (Hz)')
            plot.ylabel('Power spectrum (µV**2)')
            fig.savefig(folder+'/FR_Spectrum_'+key+addon+'.svg', transparent=True)
            plot.close()
            fig.clear()


        ###################################
        if 'CrossCorrelation' in params['Analysis'] and params['Analysis']['CrossCorrelation'] and 'spikes' in rec:
            print('CrossCorrelation')
            # cross-correlation
            scores[7] = acc(params, data.spiketrains, bin_size=params['Analysis']['CrossCorrelation']['bin_size'])
            print("CC:", scores[7])


        ###################################
        if 'LFP' in params['Analysis'] and params['Analysis']['LFP'] and 'v' in rec and 'gsyn_exc' in rec:
            print('LFP')
            lfp = LFP(data)
            vm = data.filter(name = 'v')[0]
            fig = plot.figure()
            plot.plot(lfp)
            fig.savefig(folder+'/LFP_'+key+addon+'.svg', transparent=True)
            plot.close()
            fig.clear()
            # # spectrum
            # fig = plot.figure()
            # Fs = 1 / params['dt']  # sampling frequency
            # plot.title("Spectrum")
            # # plot.magnitude_spectrum(lfp, Fs=Fs, scale='dB', color='red')
            # plot.psd(lfp, Fs=Fs)
            # fig.savefig(folder+'/Spectrum_'+key+addon+'.png')
            # fig.clear()


        # Figure( *panels ).save(folder+'/'+key+addon+".png")
        # Figure( *panels ).save(folder+'/'+key+addon+".svg")

        # for systems with low memory :)
        if removeDataFile:
            os.remove(folder+'/'+key+addon+'.pkl')

        print("scores",key,":",scores)

    return scores # to fix: is returning only the last scores!




###############################################
# ADDITIONAL FUNCTIONS



def select_spikelist( spiketrains, edge=None, limits=None ):
    new_spiketrains = []
    for i,st in enumerate(spiketrains):

        # reject cells outside limits
        if edge and limits:
            # use st entries as indexes to put ones
            x = int(i % edge)
            y = int(i / edge)
            # print(i, x,y)
            # if (x>10 and x<50) and (y>10 and y<54):
            if (x>limits[0][0] and x<limits[0][1]) and (y>limits[1][0] and y<limits[1][1]):
                # print('taken') 
                new_spiketrains.append( st )
        # new_spiketrains.append( st )
        
    return new_spiketrains



def phase_coherence(s1, s2, folder, addon, lowcut=.01, highcut=.5, fs=10, order=1):
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
    from scipy.signal import hilbert, butter, filtfilt

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    s1 = butter_bandpass_filter(s1,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    s2 = butter_bandpass_filter(s2,lowcut=lowcut,highcut=highcut,fs=fs,order=order)

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



def acc( params, spiketrains, bin_size=10, auto=False ):
    """
    Binned-time Cross-correlation
    """
    if spiketrains == [] :
        return NaN
    # create bin edges based on number of times and bin size
    # binning absolute time, and counting the number of spike times in each bin
    bin_edges = np.arange( 0, params['run_time'], bin_size )
    # print("bin_edges",bin_edges.shape,bin_edges)
    CC = []
    for i, spike_times_i in enumerate(spiketrains):
        for j, spike_times_j in enumerate(spiketrains):
            if auto and i!=j:
                continue
            itrain = np.histogram( spike_times_i, bin_edges )[0] # spike count of bin_size bins
            jtrain = np.histogram( spike_times_j, bin_edges )[0]
            if auto:
                CC.append( np.correlate(itrain, jtrain, mode="full") )
            else:
                CC.append( np.corrcoef(itrain, jtrain)[0,1] )
    if auto:
        # return np.sum(CC, axis=0)
        return CC
    else:
        return np.nanmean(CC)



def rate( params, spiketrains, bin_size=10 ):
    """
    Population rate
    as in https://neuronaldynamics.epfl.ch/online/Ch7.S2.html
    """
    if spiketrains == [] :
        return NaN
    # create bin edges based on run_time (ms) and bin size
    bin_edges = np.arange( 0, params['run_time'], bin_size )
    # print("bin_edges",bin_edges.shape)
    # binning total time, and counting the number of spike times in each bin
    hist = np.zeros( bin_edges.shape[0]-1 )
    for spike_times in spiketrains:
        hist = hist + np.histogram( spike_times, bin_edges )[0]
    return ((hist / len(spiketrains) ) / bin_size ) * 1000 # average over population: result in ms *1000 to have it in sp/s



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




def v_nullcline(x,p,I):
    gL = p['cm'] / p['tau_m']
    return -gL*(x-p['v_rest']) + gL*p['delta_T']*np.exp((x-p['v_thresh'])/p['delta_T']) + I # Touboul Brette 2008



def w_nullcline(x,p,I):
    return p['a']*(x-p['v_rest']) # Touboul Brette 2008



def nullcline(f, params, I, limits, steps): 
    p = params['Populations']['cell']['cellparams']
    fn = []
    c = np.linspace( limits[0], limits[1], steps ) # linearly spaced numbers
    for i in c:
        fn.append( f(i,p,I) )
    return c, fn


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
