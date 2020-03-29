"""Helper module with useful functions to convert raw audio in .wav format to a spectrogram
format able to be fed into the neural network. These functions, in combination with
intermediary code, form the audio preprocessing pipeline"""

def load_and_get_stats(filename):
    """Reads .wav file and returns data, sampling frequency, and length (time) of audio clip."""
    import scipy.io.wavfile as siow
    sampling_rate, amplitude_vector = siow.read(filename)

    wav_length = amplitude_vector.shape[0] / sampling_rate
    print(f"Sampling rate: {sampling_rate}")
    return sampling_rate, amplitude_vector, wav_length
 
def plot_wav_curve(filename, sampling_rate, amplitude_vector, wav_length):
    """Plots amplitude curve for a particular audio clip."""
    import matplotlib.pyplot as plt
    import numpy as np
    time = np.linspace(0, wav_length, amplitude_vector.shape[0])

    plt.plot(time, amplitude_vector)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'{filename} - viewed at {sampling_rate} samples/sec')
    plt.show()

def split_audio_into_chunks(sampling_rate, amplitude_vector, chunk_size):
    """Reshape data (amplitude vector) into many chunks of chunk_size miliseconds. Returns reshaped data and leftover data not grouped."""
    
    col_size = int(chunk_size / ((1 / sampling_rate) * 1000))
    whole = int(len(amplitude_vector) / col_size)
    first_partition_index = whole*col_size
    first_partition = amplitude_vector[:first_partition_index]
    second_partition = amplitude_vector[first_partition_index:]
    return first_partition.reshape((whole, col_size)), second_partition

def generate_spectrogram(sampling_rate, amplitude_vector):
    """Apply fourier transform to chunked audio snippets to break up each chunk into vector of scores for each frequency band. Aggregates score vectors for each snippet into spectrogram to be fed into neural network."""
    
    from scipy import signal
    import matplotlib.pyplot as plt
    import numpy as np

    frequencies, times, spectrogram = signal.spectrogram(amplitude_vector, sampling_rate)
    spectrogram = np.reshape(spectrogram, (92, -1))
    print(spectrogram.shape)
    plot_spectrogram(spectrogram)
    return spectrogram

def plot_spectrogram(spectrogram):
    """Plots spectrogram as an image."""
    import matplotlib.pyplot as plt

    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)

def fine_grain_fft(sampling_frequency, begin, end, amplitude_vector):
    """Calculates fine grain fourier transform based on sampling frequncy range specified by the user. Numpy implementation instead of scipy to allow for more user control."""
    import numpy as np
    
    sampling_interval = 1 / sampling_frequency
    time = np.arange(begin, end, sampling_interval)

    fourier_transform = np.fft.fft(amplitude_vector)
    fourier_transform = fourier_transform[range(int(len(amplitude_vector)/2))]

    return fourier_transform

def validate_fft(fourier_transform, amplitude_vector, sampling_frequency):
    """Validates fourier transform calculated by the above function by plotting graph as an image."""
    import numpy as np
    import matplotlib.pyplot as plt

    tpc = len(amplitude_vector)
    values = np.arange(int(tpc/2))
    period = tpc / sampling_frequency
    frequencies = values / period

    plt.plot(frequencies, abs(fourier_transform))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.show()

if __name__ == '__main__':
    pass