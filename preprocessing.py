import numpy as np
from scipy import signal
from scipy.signal import resample


def highpass_filter(data, cutoff=1.6, fs=100, order=5):
    """Apply high-pass filter at 1.6 Hz"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data)


def lowpass_filter(data, cutoff=45, fs=100, order=5):
    """Apply low-pass filter at 45 Hz (below Nyquist of 50 Hz)"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.95  # Use 95% of Nyquist as safety margin
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def notch_filter(data, notch_freq=48, fs=100, quality=30):
    """Apply notch filter at 48 Hz (powerline removal, adjusted for 100Hz sampling)"""
    # Ensure notch frequency is well below Nyquist
    nyquist = 0.5 * fs
    if notch_freq >= nyquist * 0.98:  # Too close to Nyquist
        notch_freq = nyquist * 0.95  # Use 95% of Nyquist
    b, a = signal.iirnotch(notch_freq, quality, fs)
    return signal.filtfilt(b, a, data)


def resample_to_100hz(data, original_fs=173.61, target_fs=100):
    """Resample data to 100 Hz"""
    num_samples = int(len(data) * target_fs / original_fs)
    resampled_data = resample(data, num_samples)
    return resampled_data


def apply_filters(data, fs=100):
    # High-pass filter
    data = highpass_filter(data, cutoff=1.6, fs=fs)
    
    # Low-pass filter (45 Hz, below Nyquist)
    data = lowpass_filter(data, cutoff=45, fs=fs)
    
    # Notch filter (48 Hz, below Nyquist)
    data = notch_filter(data, notch_freq=48, fs=fs)
    
    return data


def segment_signal(data, segment_length=100, overlap=0):
    segments = []
    step = segment_length - overlap
    
    for start in range(0, len(data) - segment_length + 1, step):
        segment = data[start:start + segment_length]
        segments.append(segment)
    
    return np.array(segments)


def minmax_scale(segment):
    
    min_val = np.min(segment)
    max_val = np.max(segment)
    
    if max_val - min_val == 0:
        return np.zeros_like(segment)
    
    return (segment - min_val) / (max_val - min_val)


def preprocess_eeg_file(file_path, original_fs=173.61, target_fs=100, 
                        segment_length=100, apply_filtering=True):
    
    # Read data
    data = np.loadtxt(file_path)
    
    # Resample to 100 Hz
    if original_fs != target_fs:
        data = resample_to_100hz(data, original_fs, target_fs)
    
    # Apply filters
    if apply_filtering:
        data = apply_filters(data, fs=target_fs)
    
    # Segment
    segments = segment_signal(data, segment_length=segment_length, overlap=0)
    
    # Min-Max scale each segment
    scaled_segments = np.array([minmax_scale(seg) for seg in segments])
    
    return scaled_segments


def preprocess_dataset(file_paths, original_fs=173.61, target_fs=100, 
                       segment_length=100, apply_filtering=True):
    all_segments = []
    file_indices = []
    
    for file_idx, file_path in enumerate(file_paths):
        segments = preprocess_eeg_file(file_path, original_fs, target_fs, 
                                       segment_length, apply_filtering)
        all_segments.append(segments)
        file_indices.extend([file_idx] * len(segments))
    
    return np.vstack(all_segments), np.array(file_indices)
