#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import librosa.display
import librosa

# %%
dataFile = h5py.File('/storage/ligo/data/data.hdf5', 'r')

strain = dataFile['strain']['Strain'].value
ts = dataFile['strain']['Strain'].attrs['Xspacing']


metaKeys = dataFile['meta'].keys()
meta = dataFile['meta']
for key in metaKeys:
    print(key, meta[key].value)

#%%
gpsStart = meta['GPSstart'].value
duration = meta['Duration'].value
gpsEnd   = gpsStart + duration

time = np.arange(gpsStart, gpsEnd, ts)

ref_time = 1187008882.4
ref_index = np.argmin(abs(time-ref_time))

padd_time = 2
padd_index = int(padd_time/ts)

event_time_slice = time[ref_index-padd_index:ref_index+padd_index]
event_strain_slice = strain[ref_index-padd_index:ref_index+padd_index]
# %%
plt.plot(event_strain_slice)
# %%
fs = 4096
NFFT = int(fs/16.0)
# and with a lot of overlap, to resolve short-time features:
NOVL = int(NFFT*15/16.0)
# choose a window that minimizes "spectral leakage" 
# (https://en.wikipedia.org/wiki/Spectral_leakage)
window = np.blackman(NFFT)

# Plot the H1 whitened spectrogram around the signal
plt.figure(figsize=(10,6))
spec_H1, freqs, bins, im = plt.specgram(event_strain_slice, NFFT=NFFT, Fs=fs, window=window, 
                                        noverlap=NOVL)

#%%

C = librosa.cqt(event_strain_slice, sr=fs/ts)
#f = librosa.cqt_frequencies(C.shape[0],fmin=64)


librosa.display.specshow(np.log10(C),
                          sr=fs/ts, x_axis='time', y_axis='cqt_hz')


# %%
