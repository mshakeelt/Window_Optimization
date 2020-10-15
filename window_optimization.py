import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.io.wavfile as wav
from sound import *


N = 8 # number of subbands
L = 32 # length of filter
n = np.arange(L)
wc = (1.0/N)*np.pi # lowpass bandwidth

ideal_low = np.sin(0.09*wc*(n-(L-1)/2))/(wc*(n-(L-1)/2)) #for 8 subbands
#ideal_low = np.sin(0.08*np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))
def errfunc(w):
    #desired passband:
   L_Pass = ideal_low * w 
   f_norm, H = sig.freqz(L_Pass, worN=512)   # frequency response
   #H = 20*np.log10(np.abs(H)+1e-6)   # frequency response in dB
    
   numfreqsamples=512
   pb=int(numfreqsamples/N)
   tb=int(numfreqsamples/(2*N))
   #H_desired=np.concatenate((np.zeros(pb), -60*np.ones(numfreqsamples-pb)))
   H_desired = np.concatenate((np.ones(pb), np.zeros(numfreqsamples-pb)))
   weights = np.concatenate((np.ones(pb), np.zeros(tb), 350*np.ones(numfreqsamples-pb-tb)))
   err = np.sum(np.abs(H-H_desired)*weights)
   print(err)
   return err

window = np.random.rand(L)
minout=opt.minimize(errfunc,window, options={'maxiter': 1000})  # get w when error is minimal
print(minout)
kw = minout.x

plt.plot(kw)
plt.title('Impulse response of Optimized Window')
plt.xlabel('Time')
plt.ylabel("Magnitude")
plt.show()

kw_w,kw_H = sig.freqz(kw)    # frequency response of FFT filters
plt.plot(kw_w,20*np.log10(np.abs(kw_H)+1e-6))
plt.title('Frequency response of Optimized Window')
plt.xlabel('Normalized Frequency')
plt.ylabel("Magnitude (dB)")
plt.show()



filt = np.zeros((N, L))
filt0 = np.sin(0.09*wc*(n-(L-1)/2))/(wc*(n-(L-1)/2)) #for 8 subbands
#filt0 = np.sin(0.08*np.pi*(n-(L-1)/2))/(np.pi*(n-(L-1)/2))
# center is wc*(i+0.5)
for i in range(0,N):
   if 0<i & i<N-1:
      filt[i] = filt0*kw*np.cos(n*wc*(i+0.5))
   elif i==0:
      filt[i] = filt0*kw
   elif i==N-1:
      filt[i] = filt0*kw*np.cos(n*np.pi)

for i in range(N):
  w,H = sig.freqz(filt[i], whole = True)    # frequency response of FFT filters
  plt.plot(w,20*np.log10(np.abs(H)+1e-6))

plt.title('Frequency response of filters')
plt.legend(('filter1','filter2','filter3','filter4','filter5','filter6','filter7','filter8'))
plt.xlabel('Normalized frequency')
plt.ylabel("Magnitude (dB)")
plt.show()

# reading audio
fs, s = wav.read("Track32.wav")
s1 = s[:,0]   #channel 1

# Analysis filter bank
filtered1 = np.zeros((N,len(s1)))
for i in range(N):
   filtered1[i] = sig.lfilter(filt[i],1,s1)

# downsampling

ds = np.zeros((N,int(len(s1)/N)))
for i in range(N):
   ds[i] = filtered1[i,0::N]

#Synthesis filter bank
# upsampling
us = np.zeros((N,len(s1)))
for i in range(N):
   us[i,0::N] = ds[i]

# filtering
filtered2 = np.zeros((N,len(s1)))
for i in range(N):
   filtered2[i] = sig.lfilter(filt[i],1,us[i])

# reconstructed signal
#sys = filtered2[0]+filtered2[1]+filtered2[2]+filtered2[3]+filtered2[4]+filtered2[5]+filtered2[6]+filtered2[7]
sys = 0
for i in range(N):
    sys += filtered2[i]

sys_w,sys_H = sig.freqz(sys, whole=True)    # frequency response of FFT filters
plt.plot(sys_w,20*np.log10(np.abs(sys_H)+1e-6))
plt.title('Frequency response of Synthesized Signal')
plt.xlabel('Normalized Frequency')
plt.ylabel("Magnitude (dB)")
plt.show()

s1_w,s1_H = sig.freqz(s1, whole=True)    # frequency response of FFT filters
plt.plot(s1_w,20*np.log10(np.abs(s1_H)+1e-6))
plt.title('Frequency response of Original Signal')
plt.xlabel('Normalized Frequency')
plt.ylabel("Magnitude (dB)")
plt.show()



#sound(sys, fs)

s1 = s1/max(s1)
sys = sys/max(sys)
l1, = plt.plot(s1, color='red')
l2, = plt.plot(sys, color='blue')
plt.legend(handles = [l1,l2,], labels = ['Original','Reconstructed'])
plt.title('Original signal and reconstructed signal')
plt.show()


