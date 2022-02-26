#
# Independent Component Analysis (ICA)
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import sys
import os
from sklearn.decomposition import FastICA, PCA
#
# Configuration
# np.set_printoptions(threshold=sys.maxsize)
np.random.seed(10)
inp_folder = 'audio'
outp_folder = 'output'
# 
# Read audio samples
s1, fs1 = sf.read(os.path.join(inp_folder, 's1.wav'))
s2, fs2 = sf.read(os.path.join(inp_folder, 's2.wav'))
#
# Reduce to shorter sample
max_len = 0
if len(s1) < len(s2):
    max_len = len(s1)
else:
    max_len = len(s2)
# Concatenate audio     
S = np.c_[s1[:max_len,1], s2[:max_len,1] ]
#
# Normalize data + add noise
# S += 0.2 * np.random.normal(size=S.shape) 
S /= S.std(axis=0)
#
# Mix data with mixing matrix (S = A * X)
A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
X = np.dot(S, A.T)  
#
# Compute ICA
ica = FastICA(n_components=2)
# Reconstruct signals
S_ = ica.fit_transform(X)  
# Get estimated mixing matrix
A_ = ica.mixing_  
# Prove ICA model by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
#
# Reconstruct signal amplitude for exported files
max_amp_s1 = max(s1[:max_len,1])
max_amp_s2 = max(s2[:max_len,1])
max_amp_s1_ica = max(S_[:,0])
max_amp_s2_ica = max(S_[:,1])
# Calculate factor
factor_s1 = max_amp_s1 / max_amp_s1_ica
factor_s2 = max_amp_s2 / max_amp_s2_ica
#
# Save reconstructed signals
sf.write(os.path.join(outp_folder, 's1_rec.wav'), S_[:,0] * factor_s1, fs1)
sf.write(os.path.join(outp_folder, 's2_rec.wav'), S_[:,1] * factor_s2, fs2)
# 
# Plot results
plt.figure()
#
models = [X, S, S_]
names = [
    "Observations (mixed signal)",
    "True Sources",
    "ICA recovered signals",
]
colors = ["red", "steelblue"]
#
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)
#
plt.tight_layout()
plt.show()