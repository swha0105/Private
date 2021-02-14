#%%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
#%%
y, sr = librosa.load(librosa.util.example_audio_file())
C = np.abs(librosa.cqt(y, sr=sr))
f = librosa.cqt_frequencies(C.shape[0],fmin=64)



librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                          sr=sr, x_axis='time', y_axis='log')

plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.tight_layout()
plt.show()

#%%

test = C 
test[1:35,1:10]  = 0

librosa.display.specshow(np.log10(test),sr=sr, x_axis='time', y_axis='log')

#%%

# %%
