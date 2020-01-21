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
    print("analysing data")
    # populations key-recorders match
    populations = {}
    for popKey,popVal in params['Populations'].items():
        if popKey != 'ext':
            if popKey in params['Recorders']:
                populations[popKey] = list(params['Recorders'][popKey].keys())
                # print(popKey, populations[popKey])


    # analysis dependent on the static values of parameters, not the running itself
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
        fig.savefig(folder+'/H_'+addon+'.svg')
        fig.clf()
        plot.close()


    # iteration over populations and selective plotting based on params and available recorders
    for key,rec in populations.items():
        # print(key)

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
        if params['Analysis']['Vm'] and 'v' in rec:
            vm = data.filter(name = 'v')[0]
            # print(vm)
            # panels.append( Panel(vm, ylabel="Membrane potential (mV)", xlabel="Time (ms)", xticks=True, yticks=True, legend=None) )
            ###################################
            # workaround
            fig = plot.figure()
            plot.plot(vm,linewidth=2)
            # plot.xlim([15000,40000])
            plot.ylim([-100,0.0]) # to plot it nicely
            # plot.ylim([-100,0.0])
            fig.savefig(folder+'/vm_'+key+addon+'.svg')
            # plot.xlim([30000,40000])
            # fig.savefig(folder+'/zoom_vm_'+key+addon+'.svg')
            plot.close()
            fig.clf()
            #################
            # Vm histogram
            fig = plot.figure()
            ylabel = key
            n,bins,patches = plot.hist(np.mean(vm,1), bins=50, normed=True) # 50*dt = 5ms bin
            fig.savefig(folder+'/Vm_histogram_'+key+addon+'.svg')
            plot.close()
            fig.clear()


        ###################################
        if params['Analysis']['PhaseSpace'] and 'w' in rec and 'v' in rec:
            vm = data.filter(name = 'v')[0]
            w = data.filter(name = 'w')[0]
            # plot W
            fig = plot.figure()
            plot.plot(w,linewidth=2)
            # plot.ylim([-100,-20.0]) # just to plot it nicely
            # plot.ylim([-100,0.0])
            fig.savefig(folder+'/w_'+key+addon+'.svg')
            fig.clf()
            plot.close()
            # plot Phase space
            I = 0 # at rest
            xn1, xn2 = nullcline( v_nullcline, params, I, (-100,0), 100 )
            I = params['Injections']['cell']['amplitude'][0]
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
            fig.savefig(folder+'/phase_'+key+addon+'.svg')
            plot.close()
            fig.clf()


        # if 'gsyn_exc' in rec:
        #     gsyn_exc = data.filter(name="gsyn_exc")[0]
        #     # panels.append( Panel(gsyn_exc,ylabel = "Exc Synaptic conductance (uS)",xlabel="Time (ms)", xticks=True,legend = None) )


        # if 'gsyn_inh' in rec:
        #     gsyn_inh = data.filter(name="gsyn_inh")[0]
        #     # panels.append( Panel(gsyn_inh,ylabel = "Inh Synaptic conductance (uS)",xlabel="Time (ms)", xticks=True,legend = None) )


        ###################################
        if params['Injections']:
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
        if params['Analysis']['Rasterplot'] and 'spikes' in rec:
            # panels.append( Panel(data.spiketrains, xlabel="Time (ms)", xticks=True, markersize=1) )
            ###################################
            # workaround
            fig = plot.figure()
            for row,st in enumerate(data.spiketrains):
                plot.scatter( st, [row]*len(st), marker='o', edgecolors='none', s=0.2 )
            fig.savefig(folder+'/spikes_'+key+addon+'.svg')
            plot.close()
            fig.clf()


        ###################################
        if params['Analysis']['ISI'] and 'spikes' in rec:
            # Spike Count
            if hasattr(data.spiketrains[0], "__len__"):
                scores[0] = len(data.spiketrains[0])
            
            # ISI
            isitot = isi(data.spiketrains)
            # print(isitot, len(isitot), np.max(isitot), np.min(isitot))
            if isinstance(isitot, (np.ndarray)):
                if len(isitot)>1:
                    scores[1] = np.mean(isitot) # mean ISI
                    scores[2] = cv(data.spiketrains) # CV

                    if key in params['Analysis']['ISI']['Populations']:
                        # Autocorrelogram
                        ac = acc(params, data.spiketrains, bin_size=params['Analysis']['ISI']['bin_size'], auto=True)
                        for i,ag in enumerate(ac):
                            x = np.linspace(-1.*(len(ag)/2), len(ag)/2, len(ag))
                            fig = plot.figure()
                            plot.plot(x,ag,linewidth=2)
                            plot.ylabel('count')
                            plot.xlabel('Time (bin='+str(params['Analysis']['ISI']['bin_size'])+'ms)')
                            fig.savefig(folder+'/Autocorrelogram_'+key+addon+'_'+str(i)+'.svg')
                            plot.close()
                            fig.clear()

                    # ISI histogram
                    fig = plot.figure()
                    plot.loglog(range(len(isitot)), isitot)
                    plot.ylabel('count')
                    plot.xlabel('ISI (ms)')
                    fig.savefig(folder+'/ISI_histogram_'+key+addon+'.svg')
                    plot.close()
                    fig.clear()

                    if params['Analysis']['ISI#']:
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
                                fig.savefig(folder+'/ISI_interval_'+key+addon+'.svg')
                                plot.close()
                                fig.clf()


        ###################################
        if params['Analysis']['FiringRate'] and 'spikes' in rec:
            # firing rate
            fr = rate(params, data.spiketrains, bin_size=10) # ms
            scores[4] = fr.mean()
            fig = plot.figure()
            plot.plot(fr,linewidth=0.5)
            plot.title("mean firing rate:"+str(scores[4])+" spikes/s")
            # plot.ylim([.0,150.])
            fig.savefig(folder+'/firingrate_'+key+addon+'.svg')
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
            fig.savefig(folder+'/FR_Spectrum_'+key+addon+'.svg')
            plot.close()
            fig.clear()


        ###################################
        if params['Analysis']['CrossCorrelation'] and 'spikes' in rec:
            # cross-correlation
            scores[7] = acc(params, data.spiketrains, bin_size=10)
            print("CC:", scores[7])


        ###################################
        if params['Analysis']['LFP'] and 'v' in rec and 'gsyn_exc' in rec:
            lfp = LFP(data)
            vm = data.filter(name = 'v')[0]
            fig = plot.figure()
            plot.plot(lfp)
            fig.savefig(folder+'/LFP_'+key+addon+'.svg')
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

def LFP(data):
    v = data.filter(name="v")[0]
    g = data.filter(name="gsyn_exc")[0]
    # We produce the current for each cell for this time interval, with the Ohm law:
    # I = g(V-E), where E is the equilibrium for exc, which usually is 0.0 (we can change it)
    # (and we also have to consider inhibitory condictances)
    i = g*(v) #AMPA
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



def load_spikelist( filename, t_start=.0, t_stop=1. ):
    spiketrains = []
    # Data is in Neo format inside a pickle file
    # open the pickle and get the neo block
    neo_block = pickle.load( open(filename, "rb") )
    # get spiketrains
    neo_spikes = neo_block.segments[0].spiketrains
    for i,st in enumerate(neo_spikes):
        for t in st.magnitude:
            spiketrains.append( (i, t) )

    spklist = SpikeList(spiketrains, list(range(len(neo_spikes))), t_start=t_start, t_stop=t_stop)
    return spklist



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



def isi( spiketrains ):
    """
    Mean Inter-Spike Intervals for all spiketrains
    """
    x = np.zeros(1)
    for st in spiketrains:
        # print( st.magnitude )
        # print( np.diff( st.magnitude ) )
        # print( np.diff( st.magnitude ).astype(int) )
        # print( np.bincount( np.diff( st.magnitude ).astype(int) ) )
        # x.append( np.bincount( np.diff( st.magnitude ).astype(int) ) )
        y = np.bincount( np.diff( st.magnitude ).astype(int) )
        x = [sum(i) for i in zip_longest(x, y, fillvalue=0.)]
    return np.array(x)/len(spiketrains)



def cv( spiketrains ):
    """
    Coefficient of variation
    """
    ii = isi(spiketrains)
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


# def VSDI( data ):
#   v = data.filter(name="v")[0]

#     # avg vm
#     sheet_indexes = data_store.get_sheet_indexes(sheet_name=sheet, neuron_ids=analog_ids)
#     positions = data_store.get_neuron_postions()[sheet]
#     print positions.shape # all 10800


#     # #################################################
#     # # FULL MAP FRAMES - ***** SINGLE TRIAL ONLY *****
#     # #################################################
#     # # segs = spont_segs # to visualize only spontaneous activity

#     # positions = numpy.transpose(positions)
#     # print positions.shape # all 10800

#     # # take the sheet_indexes positions of the analog_ids
#     # analog_positions = positions[sheet_indexes,:]
#     # print analog_positions.shape
#     # # print analog_positions

#     # # # colorbar min=resting, max=threshold
#     # # norm = ml.colors.Normalize(vmin=-80., vmax=-50., clip=True)
#     # # mapper = ml.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
#     # # mapper._A = [] # hack to plot the colorbar http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
#     # # # ml.rcParams.update({'font.size':22})
#     # # # ml.rcParams.update({'font.color':'silver'})

#     # for s in segs:

#     #   dist = eval(s.annotations['stimulus'])
#     #   print dist['radius']
#     #   if dist['radius'] < 0.1:
#     #       continue

#     #   s.load_full()
#     #   # print "s.analogsignalarrays", s.analogsignalarrays # if not pre-loaded, it results empty in loop

#     #   for a in s.analogsignalarrays:
#     #       # print "a.name: ",a.name
#     #       if a.name == 'v':
#     #           print a.shape # (10291, 900)  (instant t, cells' vm)

#     #           for t,vms in enumerate(a):
#     #               if t/10 > 500:
#     #                   break

#     #               if t%20 == 0: # each 2 ms
#     #                   time = '{:04d}'.format(t/10)

#     #                   # # open image
#     #                   # plt.figure()
#     #                   # for vm,i,p in zip(vms, analog_ids, analog_positions):
#     #                   #   # print vm, i, p
#     #                   #   plt.scatter( p[0][0], p[0][1], marker='o', c=mapper.to_rgba(vm), edgecolors='none' )
#     #                   #   plt.xlabel(time, color='silver', fontsize=22)
#     #                   # # cbar = plt.colorbar(mapper)
#     #                   # # cbar.ax.set_ylabel('mV', rotation=270)
#     #                   # # close image
#     #                   # plt.savefig( folder+"/VSDI_spont_"+parameter+"_"+str(sheet)+"_radius"+str(dist['radius'])+"_"+addon+"_time"+time+".svg", dpi=300, transparent=True )
#     #                   # plt.close()
#     #                   # gc.collect()

#     #   s.release()
