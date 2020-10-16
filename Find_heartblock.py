# Importing necessary modues:
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

## Ask input filename
filename = input("Please enter the filename of PCG signal along with extension\n(only .wav files accepted): ")

# Loading the data:
data, fs = sf.read(filename, dtype='float32')

# Designing the band pass filter
def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.sosfilt(sos, data)
        return y

# define cutoff frequencies and order of the filter
lowcut = 20
highcut = 150
order = 6
## Normalizing the data
data_1 = data/max(data)
filtered_signal_1 = butter_bandpass_filter(data_1, lowcut, highcut, fs, order)

## Performing the rolling average function:
window = int(0.05*fs)
smooth_data_1 = np.convolve(abs(filtered_signal_1), np.ones((window,))/window, mode = "same")

## Define threshold function:
def find_threshold(smooth_data):
    threshold = np.percentile(smooth_data,75)
    return (threshold)

##Finding the peaks:
threshold_1 = find_threshold(smooth_data_1)
peaks_1 = signal.find_peaks(smooth_data_1, prominence = threshold_1, distance = 0.2*fs)

##### This function dynamically calculates the mean cardiac cycle durtion from previous beats: 
def find_threshold_duration(cycle_durations, fs):
    if len(cycle_durations)==0:
        return fs
    else:
        return np.array(cycle_durations).mean()
    
#####Defining function for locating the first S1 peak:
def locate_first_s1(filtered_data, peaks_index, fs):
    '''This function takes input filtered data and location of peaks 
    to find the location of first S1 sound using frequency domain energy'''
    for i in range (4):
        beat_1 = peaks_index[i]
        beat_2 = peaks_index[i+1]
        diff = beat_2 - beat_1
        window = int(0.075*fs)
        
        ### windowing the beats
        if beat_1<=window:
            data_1 = filtered_data[0 : beat_1+window]
        else:
            data_1 = filtered_data[beat_1-window : beat_1+window]
            
        data_2 = filtered_data[beat_2-window : beat_2+window]
        
        
        ### Fourier transformation of the windowed signal 
        X1 = np.fft.fft(data_1)
        X2 = np.fft.fft(data_2)
        
        ### Normalized Energy calculation in frequency domain 
        Y1 = sum((abs(X1)/max(abs(X1)))**2)
        Y2 = sum((abs(X2)/max(abs(X2)))**2)
        
    
        if diff <=0.40*fs and diff >=0.20*fs and Y2>Y1:
            return (i)
    return (0)

### Defining the function for Characterizing Peaks
def find_beats(filtered_data, peaks, fs):
    '''The function takes location of peaks as input and characterizes peaks and also finds missing beats'''
    beats = 0
    missing = 0
    cycle_durations = []
    peaks_dict = dict()
    
    missings = []  # missing duration
    peaks_index = peaks[0]
    missing_pos=[]
    distance = 0
    
    ### locating the first s1 peak: 
    try:
        start = locate_first_s1(filtered_data, peaks_index, fs)
    except:
        start = 0
    
    track_point = "s1"
    
    for i in range (start, len(peaks_index)-1):
        distance = peaks_index[i+1] - peaks_index[i]
        threshold_duration = find_threshold_duration(cycle_durations,fs)
        
        if track_point =="s1":
            if distance >=0.20*fs and distance<=0.40*fs:
                systole = distance
                beats = beats + 1
                track_point = "s2"
                peaks_dict[peaks_index[i]]= "s1"
            elif distance >=0.40*fs and distance <= 1.5 * threshold_duration:
                cycle_durations.append(distance)
                beats = beats +1
                peaks_dict[peaks_index[i]]= "s1"
            elif distance > 1.5* threshold_duration:
                missing = missing + 1
                missings.append(distance)
                missing_pos.append(peaks_index[i])
                peaks_dict[peaks_index[i]]= "s1"
         
        elif track_point == "s2":
            if distance >=0.30*fs and distance <= 1.1*threshold_duration:
                duration = systole + distance
                cycle_durations.append(duration)
                systole = 0
                track_point = "s1"
                peaks_dict[peaks_index[i]]= "s2"
            elif distance >=0.20*fs and distance<=0.30*fs:
                systole = systole + distance
                track_point = "s3"
                peaks_dict[peaks_index[i]]= "s2"
            elif distance > 1.1 * threshold_duration: 
                missing = missing + 1
                missings.append(distance)
                missing_pos.append(peaks_index[i])
                track_point = "s1"
                peaks_dict[peaks_index[i]]= "s2"
        
        elif track_point == "s3":
            if distance >=0.20*fs and distance <=threshold_duration:
                duration = systole + distance
                cycle_durations.append(duration)
                systole = 0
                track_point = "s1"
                peaks_dict[peaks_index[i]]= "s3"
            elif distance > threshold_duration:
                missing = missing + 1
                missings.append(distance)
                missing_pos.append(peaks_index[i])
                track_point = "s1"
                peaks_dict[peaks_index[i]]= "s3"
            
    return (beats,missing,missing_pos,peaks_dict)

def plot_results(results, data, peaks, heart_rate):
    smooth_data = data
    missing_pos = results[2]
    peaks_dict = results[3]
    
    peaks = peaks[0]
    
    for i in peaks:
        try:
            if peaks_dict[i]=="s1":
                plt.annotate("s1",(i,max(smooth_data)),size = 14, weight="bold")
                plt.scatter(i,max(smooth_data),c="green", s = 200)
            elif peaks_dict[i]=="s2":
                plt.annotate("s2",(i,max(smooth_data)),size = 14, weight="bold")
                plt.scatter(i,max(smooth_data),c="red", s = 200)
            elif peaks_dict[i]=="s3":
                plt.annotate("s3",(i,max(smooth_data)),size = 14, weight="bold")
                plt.scatter(i,max(smooth_data),c="pink", s = 200)
        except:
            pass
        
    plt.plot(smooth_data)
    for i in missing_pos:
        next_peak = np.where(peaks==i)[0][0]+1
        plt.annotate("missing beat",(i,0),size = 16,verticalalignment='bottom')
        plt.plot([i,peaks[next_peak]],[0,0],c="brown",linewidth = 7.0)
        
    if results[1]==0:
        plt.title("Signal Processing Result - Normal Signal: No Missing Heart beats (Heart Rate = %d beats/min)"%heart_rate)
    else:
        title = "Signal Processing Result - Abnormal Signal: "+ str(int(results[0]/results[1])) + " : 1 Heart Block" + " (Heart Rate = " +str(heart_rate) +" beats/min)"
        plt.title(title)
    

#### Characterizing the peaks:
results_1 = find_beats(filtered_signal_1, peaks_1,fs)

heart_rate_1 = int(results_1[0]*fs/len(filtered_signal_1)*60)

if results_1[1]==0:
    print("The given PCG signal is a Normal signal with a heart rate of %d beats/min."%heart_rate_1)

else:
    print("The given PCG signal is an abnormal signal with %d:1 Second Degree Heart Block.\nHeart rate is %d beats/min."%(int(results_1[0]/results_1[1]),heart_rate_1))


plt.figure(figsize=(14,9))
plt.suptitle("RESULTS")
plt.subplot(2,1,1)
plt.title("Unprocessed PCG Signal")
plt.plot(data_1)
plt.axhline(y=0,c="black")

plt.subplot(2,1,2)
plt.axhline(y=0, c ="black")
plot_results(results_1, filtered_signal_1, peaks_1, heart_rate_1)

plt.show()

