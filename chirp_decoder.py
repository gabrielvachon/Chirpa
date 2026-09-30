import numpy as np
from scipy import signal
import collections
from matplotlib import pyplot as plt
from chirp_utils import linear_chirp, default_settings
import sys
from unireedsolomon import RSCoder

class decoder:
    """# Chirpa decoder
    Allows decoding of chirp encoded data from:
        - data blocs
        - whole audio files
    Data type must be of type *numpy.ndarray*.
    """
    def __init__(self, rs:bool, graphical:bool=False, params:dict=default_settings):
        """
        Initializes a decoder. Creates bit models for each channel, preamble models, initializes variables and buffers according to the specified settings.
        Args:
            **rs** : *bool*
                `True`, enables Reed-Solomon error correction codes.
                `False`, Reed-Solomon is disabled.            
            **graphical** : *bool* (optional)
                `True`, allows processing of graphical data. Useful for debugging and visualizing. Computationally heavy.
                `False` (default), no graphical data will generated for this instance. Computationally lighter.
            **params** : *dict* (optional)
                Allows the user to input custom settings. The settings must be placed in a dictionary. The dictionary format must follow the documentation. Alternatively, you can follow the `default_settings` variable from `chirp_utils.py` to set the parameters.
        """
        self.default_settings = default_settings
        self._set_parameters(params)
 
        self.rs_enabled = rs
        self.graphical = graphical
        
        self.N = int(self.fs * self.T)
        self.t = np.arange(self.N) / self.fs
        self.win = np.kaiser(M=self.N, beta=self.beta)

        self.bits = np.zeros((2, 8, self.N), dtype=np.float32)
        self.h_bp = np.zeros((8, self.N_fil), dtype=np.complex64)

        self.raw_buffer = collections.deque()
        self.raw_data_queue = collections.deque()

        self.state = "standby"
        self.busy = False
        self.dyn_dec_init = False
        self.end_data_found = False
        self.decoded_msg = None

        #region BITS AND FILTERS
        # Bit models and bandpass filters creation
        channel_arrangement_data = range(self.bits.shape[1])
        for m in channel_arrangement_data:
            f_s_m = self.f2 + (self.f3 - self.f2) * m / self.bits.shape[1]  # Start frequency
            f_e_m = f_s_m + (self.f3 - self.f2) / self.bits.shape[1]   # End frequency

            self.bits[0, m] = linear_chirp(                  # Down chirp generation for each channel
                t=self.t, 
                f0=f_e_m, 
                f1=f_s_m, 
                T=self.T) * self.win
            
            self.bits[1, m] = linear_chirp(                  # Up chirp generation for each channel
                t=self.t,
                f0=f_s_m,
                f1=f_e_m,
                T=self.T) * self.win

            cut_off = (f_e_m - f_s_m) / 2               # Low-pass cutoff frequency
            h = signal.firwin(
                numtaps=self.N_fil,
                cutoff=cut_off,
                fs=self.fs
                ).astype(np.float32)

            f0_f = (f_s_m + f_e_m) / 2                  # Low-pass shift -> band-pass                             
            t_f = np.arange(len(h)) / self.fs
            self.h_bp[m] = h * np.exp(2j * np.pi * f0_f * t_f)
        #endregion

        #region PREAMBLES
        # Preamble models
        self.N_pre_start = int(self.fs * self.start_pre_duration)
        self.t_pre_start = np.arange(self.N_pre_start) / self.fs
        self.start_pre_lenght = int(self.fs * (self.start_pre_duration + self.start_pre_delay))
        self.N_start_pre_delay_lenght = int(self.fs * self.start_pre_delay)
        self.win_pre_start = np.kaiser(self.N_pre_start, beta=self.start_pre_win_beta)
        self.pre_start_idx = None
        
        cut_off = (self.f1 - self.f0) / 2               # Low-pass cutoff frequency
        h_lp_pre = signal.firwin(
            numtaps=self.N_fil,
            cutoff=cut_off,
            fs=self.fs
            ).astype(np.float32)

        f0_f = (self.f1 + self.f0) / 2               
        t_f = np.arange(len(h_lp_pre)) / self.fs
        self.h_bp_pre = h_lp_pre * np.exp(2j * np.pi * f0_f * t_f)

        self.start_pre_signal = np.zeros(self.N_pre_start, dtype=np.float32)
        channel_arrangement_pre_start = range(self.bits.shape[1])
        for n in channel_arrangement_pre_start:
            f_s_m = self.f0 + (self.f1 - self.f0) * n / self.bits.shape[1]
            f_e_m = f_s_m + (self.f1 - self.f0) / self.bits.shape[1]
            self.start_pre_signal += linear_chirp(self.t_pre_start, f0=f_s_m, f1=f_e_m, T=self.start_pre_duration) * self.win_pre_start
            self.start_pre_signal += linear_chirp(self.t_pre_start, f0=f_e_m, f1=f_s_m, T=self.start_pre_duration) * self.win_pre_start

        self.N_pre_end = int(self.fs * self.end_pre_duration)
        self.t_pre_end = np.arange(self.N_pre_end) / self.fs
        self.end_pre_lenght = int(self.fs * (self.end_pre_delay + self.end_pre_duration))
        self.N_end_pre_delay_lenght = int(self.fs * self.end_pre_delay)
        self.win_pre_end = np.kaiser(self.N_pre_end, beta=self.end_pre_win_beta)
        self.pre_end_idx = None

        self.end_pre_signal = np.zeros(self.N_pre_end, dtype=np.float32)
        channel_arrangement_pre_end = range(self.bits.shape[1])
        for n in channel_arrangement_pre_end:
            f_s_m = self.f0 + (self.f1 - self.f0) * n / self.bits.shape[1]
            f_e_m = f_s_m + (self.f1 - self.f0) / self.bits.shape[1]
            self.end_pre_signal += linear_chirp(self.t_pre_end, f0=f_s_m, f1=f_e_m, T=self.end_pre_duration) * self.win_pre_end
            self.end_pre_signal += linear_chirp(self.t_pre_end, f0=f_e_m, f1=f_s_m, T=self.end_pre_duration) * self.win_pre_end
        #endregion

        self.pre_buffer_len = int(np.ceil(max(self.N_pre_start, self.N_pre_end) / self.blocksize))
        self.pre_buffer = collections.deque(maxlen=self.blocksize * self.pre_buffer_len)

        # Arrays for visualization only
        if self.graphical:
            self.rolling_buffer0 = []
            self.rolling_buffer1 = []

            # TODO: Replace this rolling buffer with separations index array -> will take less memory 
            self.rolling_buffer_divisions = collections.deque()
            for m in range(self.bits.shape[1]):
                self.rolling_buffer0.append(collections.deque())
                self.rolling_buffer1.append(collections.deque())
            self.rolling_buffer_pre = collections.deque()
            self.data_buffer_trimmed_start_idx = None
            self.data_buffer_trimmed_end_idx = None

            # TODO: Rearrange this section so the data are directly used within the dictionary
            self.graphical_data = {
                "rolling_buffer0": None,
                "rolling_buffer1": None,
                "rolling_buffer_start_end": None,
                "rolling_buffer_divisions": None,
                "data_buffer_start_idx": None,
                "data_buffer_end_idx": None,
                "data_buffer_trimmed_start_idx": None,
                "data_buffer_trimmed_end_idx": None,
                "data_buffer_bit_range": None,
                "data_buffer_bit_separation": None
                }

    def _corr_to_bytes(self, corr_data, fs, T, s_idx:int, e_idx:int, min_range:float, max_range:float):
        """
        Converts bit correlation arrays to readable bytes.
        Args:
            **corr_data** : *array*
                Array must be of size `(2, 8)`. Corresponds to correlation of symbols `0` and `1` for the `8` different channels.
            **fs** : *float*, *int*
                Sampling frequency.
            **T** : : *float*, *int*
                Period (in seconds) of a chirp.
            **s_idx** : *int*
                Start index of the data buffer.
            **e_idx** : *int*
                End index of the data buffer.
            **min_range** : *float*
                Minimum starting range. Used for the starting bound of the correlation for each chirp. E.g.: `0.1` will result in starting the correlation evaluation to 10% of the chirp lenght. 
            **max_range** : *float*
                Maximum starting range. Used for the ending bound of the correlation for each chirp. E.g.: `0.9` will result in stopping the correlation evaluation to 90% of the chirp lenght.
        Returns:
            **decoded_bytes** : *bytes*
                The decoded bytes from the correlation array. Includes Reed-Solomon codes if enabled.
            **msg_len** : *int*
                The lenght of the decoded message. Lenght includes Reed-Solomon codes if enabled.
        """
        decoded = []
        N_bit = int(fs * T)
        min_threshold = int(N_bit * min_range)
        max_threshold = int(N_bit * (1 - max_range))
        bits = corr_data.shape[0]
        channels = corr_data.shape[1]
        
        # Trims the correlation data to get only the data part. The trimming works based on the preamble correlation peak. 
        # We assume that the maximum correlation occurs at {preamble duration}/2 and we use it as a starting point.
        # We also need to include the delay between the preamble and the data buffer. Since the data buffer doesn't start 
        # filling right when the preamble peak occurs, we need to compensate for the time between the peak and the start 
        # of the data buffer, which is the position of the preamble peak of the previous buffer.
        corr_data_trimmed = corr_data[:, :, s_idx:e_idx]
        data_len = corr_data_trimmed.shape[2]
        msg_len = int(np.round(data_len / N_bit))
        idx_start = 0
        last_idx = data_len - 1
        for i in range(msg_len):
            byte_array = []
            for n in range(channels):
                rand = False
                idx_start = i * N_bit
                idx_end = min(idx_start + N_bit, last_idx)

                if idx_start + N_bit < last_idx:
                    range_zero = corr_data_trimmed[0][n][idx_start+min_threshold:idx_end-max_threshold]
                    range_one = corr_data_trimmed[1][n][idx_start+min_threshold:idx_end-max_threshold]
                else:
                    range_zero = corr_data_trimmed[0][n][idx_start+min_threshold:last_idx-max_threshold]
                    range_one = corr_data_trimmed[1][n][idx_start+min_threshold:last_idx-max_threshold]
                
                if self.cand_win:
                    range_zero *= np.kaiser(M=len(range_zero), beta=self.cand_beta_win)
                    range_one *= np.kaiser(M=len(range_one), beta=self.cand_beta_win)

                if self.candidate_selection_mode == "max":
                    if np.max(range_zero) > np.max(range_one): #if RMS_zero > RMS_one:
                        byte_array.append(0)
                    elif np.max(range_zero) < np.max(range_one): #RMS_zero < RMS_one:
                        byte_array.append(1)
                    else:
                        rand = True
                elif self.candidate_selection_mode == "rms":
                    if idx_start + N_bit < last_idx:
                        RMS_zero = np.sqrt(1 / N_bit * np.sum(range_zero ** 2))
                        RMS_one = np.sqrt(1 / N_bit * np.sum(range_one ** 2))
                    else:
                        RMS_zero = np.sqrt(1 / (last_idx - idx_start) * np.sum(range_zero ** 2))
                        RMS_one = np.sqrt(1 / (last_idx - idx_start) * np.sum(range_one ** 2))
                    if RMS_zero > RMS_one:
                        byte_array.append(0)
                    elif RMS_zero < RMS_one:
                        byte_array.append(1)
                    else:
                        rand = True
                if rand:
                    byte_array.append(np.random.choice([0, 1]))
                    print("rand")
            bits_str = ''.join(str(b) for b in byte_array[::-1])
            byte = int(bytes(bits_str, 'ascii'), 2)
            decoded.append(byte)
            idx_start += N_bit
        decoded_bytes = bytes(decoded)
        return decoded_bytes, msg_len

    def _get_peaks(self, NCC:np.array, fs, T, height, width, threshold, distance, prominence):
        """
        Deprecated
        """
        if NCC.shape != (2, 8, NCC.shape[2]):
            raise(ValueError(f"Array must be of shape (2, 8, {NCC.shape[2]})"))
        
        bit_0 = []
        bit_1 = []

        for i in range(NCC.shape[1]):
            bit_0.append(signal.find_peaks(x=NCC[0][i], height=height, distance=int(distance*fs*T), width=width, threshold=threshold, prominence=prominence))
            bit_1.append(signal.find_peaks(x=NCC[1][i], height=height, distance=int(distance*fs*T), width=width, threshold=threshold, prominence=prominence))

        return bit_0, bit_1

    def decode(self, data):
        """# Decode
        Allows to decode a Chirpa encoded message.
        Args:
            **data** : *numpy.ndarray*
                The audio data. Must be a 1-D numpy array.
                The audio signal received in either:
                    - **blocks** (dynamic mode) : For realtime processing, see documentation for implementation example. The array must be of length `blocksize`. 
                    - **full lenght** (static mode) : For audio file processing. Array length can be as long as desired.
        Returns:
            **message** : *bytes*
                The decoded message as byte array.
        """
        if len(data) > self.blocksize:
            if not self.busy:
                return self._static_decode(data)
            print("Decoder is busy...")
        else:
            return self._dynamic_decode(data)

    def _static_decode(self, data):
        """
        Subdecoding function that decodes an audio data array independently of its size.
        Args:
            **data** : *numpy.ndarray*
                The audio data. Must be a 1-D numpy array.
        """
        msg = None
        msg_len = None
        self.decoded_msg = None
        
        pre_buffer_len = int(np.ceil(max(self.N_pre_start, self.N_pre_end) / self.blocksize)) # added ceil
        self.raw_data_queue.clear()

        self.pre_buffer = collections.deque(maxlen=self.blocksize * pre_buffer_len)
        self.pre_buffer.extend(np.zeros(self.blocksize * pre_buffer_len, dtype=np.complex64))
        pre_start_idx = 0
        pre_end_idx = 0

        s_range = 0
        e_range = self.blocksize

        end_data_found = False
        while e_range < len(data):
            e_range = s_range + self.blocksize
            buf = data[s_range:e_range]
            self.pre_buffer.extend(signal.fftconvolve(in1=buf, in2=self.h_bp_pre, mode="same"))

            if self.state == "standby":
                raw_corr_start_pre = signal.correlate(in1=self.pre_buffer, in2=self.start_pre_signal, method='fft', mode='same').astype(np.complex64)
                norm_local = np.sqrt(np.sum(np.abs(self.pre_buffer) ** 2)) * np.sqrt(np.sum(np.abs(self.start_pre_signal) ** 2))      # normalized cross-correlation
                norm_corr_start_pre = np.abs(raw_corr_start_pre) / norm_local

                max_corr_start_pre = np.max(np.abs(norm_corr_start_pre))
                if max_corr_start_pre > self.start_pre_threshold:
                    self.pre_buffer.clear()
                    for i in range(0, len(norm_corr_start_pre), self.blocksize):
                        if max_corr_start_pre in norm_corr_start_pre[i:i+self.blocksize]:
                            if self.graphical:
                                self.rolling_buffer_pre.extend(norm_corr_start_pre[i:i+self.blocksize])
                            pre_start_idx = np.argmax(norm_corr_start_pre[i:i+self.blocksize])
                    if self.graphical:
                        for n in range(self.bits.shape[1]):
                            self.rolling_buffer0[n].extend(np.zeros(self.blocksize))
                            self.rolling_buffer1[n].extend(np.zeros(self.blocksize))
                        division_idx = np.zeros(self.blocksize)
                        division_idx[0] = 1
                        self.rolling_buffer_divisions.extend(np.zeros(self.blocksize))
                        self.rolling_buffer_divisions.extend(division_idx)
                    self.state="acq"

            elif self.state == "acq":
                self.raw_data_queue.append(buf)

                raw_corr_end_pre = signal.correlate(in1=self.pre_buffer, in2=self.end_pre_signal, method='fft', mode='same').astype(np.complex64)
                norm_local = np.sqrt(np.sum(np.abs(self.pre_buffer) ** 2)) * np.sqrt(np.sum(np.abs(self.end_pre_signal) ** 2))
                norm_corr_end_pre = np.abs(raw_corr_end_pre) / norm_local

                if self.graphical:
                    division_idx = np.zeros(self.blocksize)
                    division_idx[0] = 1
                    self.rolling_buffer_divisions.extend(division_idx)
                    self.rolling_buffer_pre.extend(norm_corr_end_pre[-self.blocksize:])

                if len(self.raw_data_queue) * self.blocksize < self.N_pre_start / 2 + self.N_start_pre_delay_lenght + self.N_end_pre_delay_lenght:
                    s_range = e_range
                    continue

                max_corr_end_pre = np.max(np.abs(norm_corr_end_pre[int(self.N_pre_start / 2 + self.N_start_pre_delay_lenght + self.N_end_pre_delay_lenght):])) # edit, added range in array to prevent max value from start preamble from detecting false end preamble positive

                if max_corr_end_pre > self.end_pre_threshold:
                    for i in range(0, len(norm_corr_end_pre), self.blocksize):
                        if max_corr_end_pre in norm_corr_end_pre[i:i+self.blocksize]:
                            pre_end_idx = np.argmax(norm_corr_end_pre[i:i+self.blocksize])
                            end_data_found = True
                    self.state = "decode"
            s_range = e_range

        if not end_data_found:
            raise(IndexError("End preamble not detected"))

        band_buffer = np.zeros((self.bits.shape[1], len(self.raw_data_queue), self.blocksize), dtype=np.complex64)
        block_idx = 0
        for block in self.raw_data_queue:
            for n in range(self.bits.shape[1]):
                conv = signal.fftconvolve(in1=block, in2=self.h_bp[n], mode='same')
                band_buffer[n][block_idx][0:conv.shape[0]] = conv

            block_idx += 1

        band_buffer = np.reshape(band_buffer, (self.bits.shape[1], len(self.raw_data_queue) * self.blocksize))
        corr_data = np.zeros((self.bits.shape[0], self.bits.shape[1], len(self.raw_data_queue) * self.blocksize), dtype=np.float32)

        for m in range(self.bits.shape[0]):
            for n in range(self.bits.shape[1]):
                raw_corr = signal.correlate(in1=band_buffer[n], in2=self.bits[m][n], method='fft', mode='same').astype(np.complex64)                
                norm_local = np.sqrt(np.sum(np.abs(band_buffer[n]) ** 2)) * np.sqrt(np.sum(np.abs(self.bits[m][n]) ** 2)) + 1e-12 # edit added 1e-12
                corr_data[m][n] = np.abs(raw_corr) / norm_local

        corr_data = np.sign(corr_data) * np.abs(corr_data) ** self.correlation_power_factor
        for n in range(self.bits.shape[1]):
            max_corr_bit_0 = np.max(np.abs(corr_data[0][n]))
            max_corr_bit_1 = np.max(np.abs(corr_data[1][n]))
            if max_corr_bit_0 > max_corr_bit_1:
                corr_data[0][n] /= max_corr_bit_0
                corr_data[1][n] /= max_corr_bit_0
            else:
                corr_data[0][n] /= max_corr_bit_1
                corr_data[1][n] /= max_corr_bit_1

        if self.graphical:
            for m in range(self.bits.shape[0]):
                for n in range(self.bits.shape[1]):
                    if m == 0:
                        self.rolling_buffer0[n].extend(corr_data[m][n])
                    else:
                        self.rolling_buffer1[n].extend(corr_data[m][n])

        try:
            data_buffer_trimmed_start_idx = int((self.start_pre_duration / 2 + self.start_pre_delay) * self.fs) - self.blocksize + pre_start_idx
            data_buffer_trimmed_end_idx = corr_data.shape[2] - int((self.end_pre_duration / 2 + self.end_pre_delay) * self.fs) - self.blocksize + pre_end_idx

            msg, msg_len = self._corr_to_bytes(corr_data=corr_data, fs=self.fs, T=self.T, s_idx=data_buffer_trimmed_start_idx, e_idx=data_buffer_trimmed_end_idx, min_range=self.min_range, max_range=self.max_range) #corr_to_char(bit_0=bit_0, bit_1=bit_1, fs=fs, T=T)

            if msg:
                print(f"Received ({msg_len} B): {msg}\n")
                sys.stdout.flush()
        except Exception as e:
            print(f"Correlation data conversion error: {e}\n")

        self.raw_data_queue.clear()
        self.state = "standby"

        if self.graphical:
            buffer_bit_range = []
            buffer_bit_separation = []
            for i in range(data_buffer_trimmed_start_idx, data_buffer_trimmed_end_idx, self.N):
                buffer_bit_separation.append(i + np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0])
            for i in range(data_buffer_trimmed_start_idx, data_buffer_trimmed_end_idx, self.N):
                buffer_bit_range.append(i + np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0] + int(self.N * self.min_range))
                buffer_bit_range.append(i + np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0] + int(self.N * self.max_range))
            self.graphical_data["rolling_buffer0"] = np.array(self.rolling_buffer0, dtype=np.float32, copy=True)
            self.graphical_data["rolling_buffer1"] = np.array(self.rolling_buffer1, dtype=np.float32, copy=True)
            self.graphical_data["rolling_buffer_pre"] = np.array(self.rolling_buffer_pre, dtype=np.float32, copy=True)
            self.graphical_data["rolling_buffer_divisions"] = np.array(self.rolling_buffer_divisions, dtype=np.float32, copy=True)
            self.graphical_data["data_buffer_start_idx"] = np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0]
            self.graphical_data["data_buffer_end_idx"] = np.argwhere(np.array(self.rolling_buffer_divisions)==1)[-1]
            self.graphical_data["data_buffer_trimmed_start_idx"] = np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0] + data_buffer_trimmed_start_idx
            self.graphical_data["data_buffer_trimmed_end_idx"] = np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0] + data_buffer_trimmed_end_idx
            self.graphical_data["data_buffer_bit_range"] = buffer_bit_range
            self.graphical_data["data_buffer_bit_separation"] = buffer_bit_separation

            for n in range(self.bits.shape[1]):
                self.rolling_buffer0[n].clear()
                self.rolling_buffer1[n].clear()

            self.rolling_buffer_pre.clear()
            self.rolling_buffer_divisions.clear()

        self.pre_buffer.clear()

        if self.rs_enabled:
            rs_dec = RSCoder(msg_len, int(np.round(msg_len/self.rs_ratio)))
            try:
                self.decoded_msg, _ = rs_dec.decode(msg)
            except Exception as e:
                print(f"Error: {e}")
        else: 
            self.decoded_msg = msg

        return self.decoded_msg

    def _dynamic_decode(self, data):
        """
        Subdecoding function that decodes a `blocksize` lenght audio data array. The state of the decoder is saved each time the processing of a block is done. This function is called repeatedly to process realtime audio.
        Args:
            **data** : *numpy.ndarray*
                The audio data. Must be a 1-D numpy array of size `blocksize`.
        """
        if not self.dyn_dec_init:
            msg = None
            msg_len = None
            self.raw_data_queue.clear()

            self.pre_buffer = collections.deque(maxlen=self.blocksize * self.pre_buffer_len)
            self.pre_buffer.extend(np.zeros(self.blocksize * self.pre_buffer_len, dtype=np.complex64))
            self.pre_start_idx = 0
            self.pre_end_idx = 0

            self.end_data_found = False
            self.busy = True
            self.dyn_dec_init = True
            self.decoded_msg = None

            if self.graphical:
                self.rolling_buffer_pre = collections.deque()
                self.rolling_buffer_divisions = collections.deque()

        buf = data
        self.pre_buffer.extend(signal.fftconvolve(in1=buf, in2=self.h_bp_pre, mode="same"))

        if self.state == "standby":
            raw_corr_start_pre = signal.correlate(in1=self.pre_buffer, in2=self.start_pre_signal, method='fft', mode='same').astype(np.complex64)
            norm_local = np.sqrt(np.sum(np.abs(self.pre_buffer) ** 2)) * np.sqrt(np.sum(np.abs(self.start_pre_signal) ** 2))      # normalized cross-correlation
            norm_corr_start_pre = np.abs(raw_corr_start_pre) / norm_local

            max_corr_start_pre = np.max(np.abs(norm_corr_start_pre))
            if max_corr_start_pre > self.start_pre_threshold:
                self.pre_buffer.clear()
                for i in range(0, len(norm_corr_start_pre), self.blocksize):
                    if max_corr_start_pre in norm_corr_start_pre[i:i+self.blocksize]:
                        if self.graphical:
                            self.rolling_buffer_pre.extend(norm_corr_start_pre[i:i+self.blocksize])
                        self.pre_start_idx = np.argmax(norm_corr_start_pre[i:i+self.blocksize])
                if self.graphical:
                    for n in range(self.bits.shape[1]):
                        self.rolling_buffer0[n].extend(np.zeros(self.blocksize))
                        self.rolling_buffer1[n].extend(np.zeros(self.blocksize))
                    division_idx = np.zeros(self.blocksize)
                    division_idx[0] = 1
                    self.rolling_buffer_divisions.extend(np.zeros(self.blocksize))
                    self.rolling_buffer_divisions.extend(division_idx)
                self.state="acq"

        # TODO : Add a timeout if data transmission is interrupted or end preamble is not detected
        elif self.state == "acq":
            self.raw_data_queue.append(buf)

            raw_corr_end_pre = signal.correlate(in1=self.pre_buffer, in2=self.end_pre_signal, method='fft', mode='same').astype(np.complex64)
            norm_local = np.sqrt(np.sum(np.abs(self.pre_buffer) ** 2)) * np.sqrt(np.sum(np.abs(self.end_pre_signal) ** 2))
            norm_corr_end_pre = np.abs(raw_corr_end_pre) / norm_local

            if self.graphical:
                division_idx = np.zeros(self.blocksize)
                division_idx[0] = 1
                self.rolling_buffer_divisions.extend(division_idx)
                self.rolling_buffer_pre.extend(norm_corr_end_pre[-self.blocksize:])

            if len(self.raw_data_queue) * self.blocksize > self.N_pre_start / 2 + self.N_start_pre_delay_lenght + self.N_end_pre_delay_lenght:
                max_corr_end_pre = np.max(np.abs(norm_corr_end_pre[int(self.N_pre_start / 2 + self.N_start_pre_delay_lenght + self.N_end_pre_delay_lenght):])) # edit, added range in array to prevent max value from start preamble from detecting false end preamble positive
                if max_corr_end_pre > self.end_pre_threshold:
                    for i in range(0, len(norm_corr_end_pre), self.blocksize):
                        if max_corr_end_pre in norm_corr_end_pre[i:i+self.blocksize]:
                            self.pre_end_idx = np.argmax(norm_corr_end_pre[i:i+self.blocksize])
                            self.end_data_found = True
                    self.state = "decode"

        elif self.state == "decode":
            band_buffer = np.zeros((self.bits.shape[1], len(self.raw_data_queue), self.blocksize), dtype=np.complex64)
            block_idx = 0

            # TODO : Try filtering with whole channel instead of individual blocks 
            for block in self.raw_data_queue:
                for n in range(self.bits.shape[1]):
                    conv = signal.fftconvolve(in1=block, in2=self.h_bp[n], mode='same')
                    band_buffer[n][block_idx][0:conv.shape[0]] = conv

                block_idx += 1

            band_buffer = np.reshape(band_buffer, (self.bits.shape[1], len(self.raw_data_queue) * self.blocksize))
            corr_data = np.zeros((self.bits.shape[0], self.bits.shape[1], len(self.raw_data_queue) * self.blocksize), dtype=np.float32)

            for m in range(self.bits.shape[0]):
                for n in range(self.bits.shape[1]):
                    raw_corr = signal.correlate(in1=band_buffer[n], in2=self.bits[m][n], method='fft', mode='same').astype(np.complex64)                    
                    norm_local = np.sqrt(np.sum(np.abs(band_buffer[n]) ** 2)) * np.sqrt(np.sum(np.abs(self.bits[m][n]) ** 2)) + 1e-12 # edit added 1e-12
                    corr_data[m][n] = np.abs(raw_corr) / norm_local

            corr_data = np.sign(corr_data) * np.abs(corr_data) ** self.correlation_power_factor
            for n in range(self.bits.shape[1]):
                max_corr_bit_0 = np.max(np.abs(corr_data[0][n]))
                max_corr_bit_1 = np.max(np.abs(corr_data[1][n]))
                if max_corr_bit_0 > max_corr_bit_1:
                    corr_data[0][n] /= max_corr_bit_0
                    corr_data[1][n] /= max_corr_bit_0
                else:
                    corr_data[0][n] /= max_corr_bit_1
                    corr_data[1][n] /= max_corr_bit_1

            if self.graphical:
                for m in range(self.bits.shape[0]):
                    for n in range(self.bits.shape[1]):
                        if m == 0:
                            self.rolling_buffer0[n].extend(corr_data[m][n])
                        else:
                            self.rolling_buffer1[n].extend(corr_data[m][n])

            try:
                self.data_buffer_trimmed_start_idx = int((self.start_pre_duration / 2 + self.start_pre_delay) * self.fs) - self.blocksize + self.pre_start_idx
                self.data_buffer_trimmed_end_idx = corr_data.shape[2] - int((self.end_pre_duration / 2 + self.end_pre_delay) * self.fs) - self.blocksize + self.pre_end_idx

                msg, msg_len = self._corr_to_bytes(corr_data=corr_data, fs=self.fs, T=self.T, s_idx=self.data_buffer_trimmed_start_idx, e_idx=self.data_buffer_trimmed_end_idx, min_range=self.min_range, max_range=self.max_range) #corr_to_char(bit_0=bit_0, bit_1=bit_1, fs=fs, T=T)

                print(f"Received ({msg_len} B): {msg}\n") and sys.stdout.flush() if msg != "" else ""
            except Exception as e:
                print(f"Correlation data conversion error: {e}\n")

            self.raw_data_queue.clear()
            self.state = "standby"

            if self.graphical:
                buffer_bit_range = []
                buffer_bit_separation = []
                for i in range(self.data_buffer_trimmed_start_idx, self.data_buffer_trimmed_end_idx, self.N):
                    buffer_bit_separation.append(i + np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0])
                for i in range(self.data_buffer_trimmed_start_idx, self.data_buffer_trimmed_end_idx, self.N):
                    buffer_bit_range.append(i + np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0] + int(self.N * self.min_range))
                    buffer_bit_range.append(i + np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0] + int(self.N * self.max_range))
                self.graphical_data["rolling_buffer0"] = np.array(self.rolling_buffer0, dtype=np.float32, copy=True)
                self.graphical_data["rolling_buffer1"] = np.array(self.rolling_buffer1, dtype=np.float32, copy=True)
                self.graphical_data["rolling_buffer_pre"] = np.array(self.rolling_buffer_pre, dtype=np.float32, copy=True)
                self.graphical_data["rolling_buffer_divisions"] = np.array(self.rolling_buffer_divisions, dtype=np.float32, copy=True)
                self.graphical_data["data_buffer_start_idx"] = np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0]
                self.graphical_data["data_buffer_end_idx"] = np.argwhere(np.array(self.rolling_buffer_divisions)==1)[-1]
                self.graphical_data["data_buffer_trimmed_start_idx"] = np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0] + self.data_buffer_trimmed_start_idx
                self.graphical_data["data_buffer_trimmed_end_idx"] = np.argwhere(np.array(self.rolling_buffer_divisions)==1)[0] + self.data_buffer_trimmed_end_idx
                self.graphical_data["data_buffer_bit_range"] = buffer_bit_range
                self.graphical_data["data_buffer_bit_separation"] = buffer_bit_separation

                for n in range(self.bits.shape[1]):
                    self.rolling_buffer0[n].clear()
                    self.rolling_buffer1[n].clear()

                self.rolling_buffer_pre.clear()
                self.rolling_buffer_divisions.clear()

            self.pre_buffer.clear()

            if self.rs_enabled:
                rs_dec = RSCoder(msg_len, int(np.round(msg_len/self.rs_ratio)))
                try:
                    self.decoded_msg, _ = rs_dec.decode(msg)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                self.decoded_msg = msg

            self.busy = False
            self.dyn_dec_init = False
            
            return self.decoded_msg

    def __str__(self):
        pass

    def show(self):
        """
        Displays a correlation graph for each channels and for the preambles.\n
        ## Color coding\n
        - **blue** : zeros correlation signals, the 8 first plots;\n
        - **orange** : ones correlation signals, the 8 first plots;\n
        - **red** : preamble correlation signal, the bottom plot;\n
        - **light gray** : blocksize separation, only for trimmed data;\n
        - **lime** : start and end of *data buffer*;\n
        - **fuchsia** : start and end index of the *trimmed data*, based on start and end preamble timing;\n
        - **pink** : start and end *range* of each bit in its delimitation;\n
        - **red** : time delimitation of `N` samples corresponding to the delimitation of a bit;\n
        """
        if self.graphical:
            fig, ax = plt.subplots(self.bits.shape[1] + 1, 1, sharex=True)
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.1, hspace=0)
            for a in ax[:self.bits.shape[1]-1]:
                a.xaxis.set_visible(False)
            rb_0 = self.graphical_data["rolling_buffer0"]                    # zeros correlation signals, the 8 first plots; color:blue
            rb_1 = self.graphical_data["rolling_buffer1"]                    # ones correlation signals, the 8 first plots; color:orange
            rb_p = self.graphical_data["rolling_buffer_pre"]                 # preamble correlation signal, the bottom plot: color:red
            rb_d = self.graphical_data["rolling_buffer_divisions"]           # blocksize separation, only for trimmed data; color:light gray
            db_s = self.graphical_data["data_buffer_start_idx"]              # start of **data buffer**; color:lime
            db_e = self.graphical_data["data_buffer_end_idx"]                # end of **data buffer**; color:lime
            db_s_t = self.graphical_data["data_buffer_trimmed_start_idx"]    # start index of the **trimmed data**, based on start preamble timing; color:fuchsia
            db_e_t = self.graphical_data["data_buffer_trimmed_end_idx"]      # end index of the **trimmed data**, based on end preamble timing; color:fuchsia
            db_br = self.graphical_data["data_buffer_bit_range"]             # start and end **range** of each bit in its delimitation; color:pink
            db_bs = self.graphical_data["data_buffer_bit_separation"]        # time delimitation of N samples corresponding to the delimitation of a bit: color:red

            for n in range(self.bits.shape[1]):
                ax[n].vlines(x=np.where(np.array(rb_d) == 1), ymin=np.min([rb_0[n], rb_1[n]]), ymax=np.max([rb_0[n], rb_1[n]]), color='whitesmoke')
                ax[n].vlines(x=[db_s, db_e], ymin=np.min([rb_0[n], rb_1[n]]), ymax=np.max([rb_0[n], rb_1[n]]), color='lime')
                ax[n].plot(np.array(rb_0[n]))
                ax[n].plot(np.array(rb_1[n]))
                ax[n].vlines(x=db_br, ymin=np.min([rb_0[n], rb_1[n]]), ymax=np.max([rb_0[n], rb_1[n]]), color='pink')
                ax[n].vlines(x=db_bs, ymin=np.min([rb_0[n], rb_1[n]]), ymax=np.max([rb_0[n], rb_1[n]]), color='red')
                ax[n].vlines(x=[db_s_t, db_e_t], ymin=np.min([rb_0[n], rb_1[n]]), ymax=np.max([rb_0[n], rb_1[n]]), color='fuchsia')
            ax[n + 1].plot(np.array(rb_p), "C3")
            plt.show()
        else:
            print(f"Note: Graphical mode = {self.graphical}.")

    # TODO : Optimize this function with loops instead of if/elses
    def _set_parameters(self, params:dict):
        """
        Take a dictionary containing the settings as input (optional). By default, default settings from the `default_settings` from `chirp_utils.py` will be used.
        """
        if type(params["chirp_settings"]["freq_range"]["f2"]) in (float, int):
            self.f2 = params["chirp_settings"]["freq_range"]["f2"]
        else:
            print(f"Invalid type, using default parameter for [\"chirp_settings\"][\"freq_range\"][\"f2\"])")
            self.f2 = self.default_settings["chirp_settings"]["freq_range"]["f2"]

        if type(params["chirp_settings"]["freq_range"]["f3"]) in (float, int):
            self.f3 = params["chirp_settings"]["freq_range"]["f3"]
        else:
            print(f"Invalid type, using default parameter for [\"chirp_settings\"][\"freq_range\"][\"f3\"])")
            self.f3 = self.default_settings["chirp_settings"]["freq_range"]["f3"]

        if type(params["sample_rate"]) in (float, int):
            self.fs = params["sample_rate"]
        else:
            print(f"Invalid type, using default parameter for [\"sample_rate\"])")
            self.fs = self.default_settings["sample_rate"]

        if type(params["chirp_settings"]["chirp_period"]) in (float, int):
            self.T = params["chirp_settings"]["chirp_period"]
        else:
            print(f"Invalid type, using default parameter for [\"chirp_settings\"][\"chirp_period\"])")
            self.T = self.default_settings["chirp_settings"]["chirp_period"]

        if type(params["chirp_settings"]["type"]) == str:
            self.type = params["chirp_settings"]["type"]
        else:
            print(f"Invalid type, using default parameter for [\"chirp_settings\"][\"type\"])")
            self.type = self.default_settings["chirp_settings"]["type"]

        #TODO : Add window type or remove
        if type(params["chirp_settings"]["window"]["beta"]) in (float, int):
            self.beta = params["chirp_settings"]["window"]["beta"]
        else:
            print(f"Invalid type, using default parameter for [\"chirp_settings\"][\"window\"][\"beta\"])")
            self.beta = self.default_settings["chirp_settings"]["window"]["beta"]

        if type(params["decoder"]["start_bit_range"]) in (float, int):
            self.min_range = params["decoder"]["start_bit_range"]
        else:
            print(f"Invalid type, using default parameter for [\"decoder\"][\"start_bit_range\"])")
            self.min_range = self.default_settings["decoder"]["start_bit_range"]

        if type(params["decoder"]["end_bit_range"]) in (float, int):
            self.max_range = params["decoder"]["end_bit_range"]
        else:
            print(f"Invalid type, using default parameter for [\"decoder\"][\"end_bit_range\"])")
            self.max_range = self.default_settings["decoder"]["end_bit_range"]

        if type(params["decoder"]["correlation_power_factor"]) in (float, int):
            self.correlation_power_factor = params["decoder"]["correlation_power_factor"]
        else:
            print(f"Invalid type, using default parameter for [\"decoder\"][\"correlation_power_factor\"])")
            self.correlation_power_factor = self.default_settings["decoder"]["correlation_power_factor"]

        if type(params["decoder"]["candidate_selection_mode"]) == str:
            self.candidate_selection_mode = params["decoder"]["candidate_selection_mode"]
        else:
            print(f"Invalid type, using default parameter for [\"decoder\"][\"candidate_selection_mode\"])")
            self.candidate_selection_mode = self.default_settings["decoder"]["candidate_selection_mode"]

        if type(params["decoder"]["candidate_window"]) == bool:
            self.cand_win = params["decoder"]["candidate_window"]
        else:
            print(f"Invalid type, using default parameter for [\"decoder\"][\"candidate_window\"])")
            self.cand_win = self.default_settings["decoder"]["candidate_window"]

        if type(params["decoder"]["candidate_window_beta"]) in (float, int):
            self.cand_beta_win = params["decoder"]["candidate_window_beta"]
        else:
            print(f"Invalid type, using default parameter for [\"decoder\"][\"candidate_window_beta\"])")
            self.cand_beta_win = self.default_settings["decoder"]["candidate_window_beta"]

        if type(params["filter_order"]) == int:
            self.N_fil = params["filter_order"]
        else:
            print(f"Invalid type, using default parameter for [\"filter_order\"])")
            self.N_fil = self.default_settings["filter_order"]

        if type(params["blocksize"]) == int:
            self.blocksize = params["blocksize"]
        else:
            print(f"Invalid type, using default parameter for [\"blocksize\"])")
            self.blocksize = self.default_settings["blocksize"]

        if type(params["preamble_settings"]["freq_range"]["f0"]) in (float, int):
            self.f0 = params["preamble_settings"]["freq_range"]["f0"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"freq_range\"][\"f0\"])")
            self.f0 = self.default_settings["preamble_settings"]["freq_range"]["f0"]

        if type(params["preamble_settings"]["freq_range"]["f1"]) in (float, int):
            self.f1 = params["preamble_settings"]["freq_range"]["f1"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"freq_range\"][\"f1\"])")
            self.f1 = self.default_settings["preamble_settings"]["freq_range"]["f1"]

        if type(params["preamble_settings"]["start"]["duration"]) in (float, int):
            self.start_pre_duration = params["preamble_settings"]["start"]["duration"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"start\"][\"duration\"])")
            self.start_pre_duration = self.default_settings["preamble_settings"]["start"]["duration"]

        if type(params["preamble_settings"]["start"]["delay"]) in (float, int):
            self.start_pre_delay = params["preamble_settings"]["start"]["delay"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"start\"][\"delay\"])")
            self.start_pre_delay = self.default_settings["preamble_settings"]["start"]["delay"]

        if type(params["preamble_settings"]["start"]["window"]["beta"]) in (float, int):
            self.start_pre_win_beta = params["preamble_settings"]["start"]["window"]["beta"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"start\"][\"window\"][\"beta\"])")
            self.start_pre_win_beta = self.default_settings["preamble_settings"]["start"]["window"]["beta"]

        if type(params["preamble_settings"]["start"]["threshold"]) in (float, int):
            self.start_pre_threshold = params["preamble_settings"]["start"]["threshold"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"start\"][\"threshold\"])")
            self.start_pre_threshold = self.default_settings["preamble_settings"]["start"]["threshold"]

        if type(params["preamble_settings"]["end"]["duration"]) in (float, int):
            self.end_pre_duration = params["preamble_settings"]["end"]["duration"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"end\"][\"duration\"])")
            self.end_pre_duration = self.default_settings["preamble_settings"]["end"]["duration"]

        if type(params["preamble_settings"]["end"]["delay"]) in (float, int):
            self.end_pre_delay = params["preamble_settings"]["end"]["delay"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"end\"][\"delay\"])")
            self.end_pre_delay = self.default_settings["preamble_settings"]["end"]["delay"]

        if type(params["preamble_settings"]["end"]["window"]["beta"]) in (float, int):
            self.end_pre_win_beta = params["preamble_settings"]["end"]["window"]["beta"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"end\"][\"window\"][\"beta\"])")
            self.end_pre_win_beta = self.default_settings["preamble_settings"]["end"]["window"]["beta"]

        if type(params["preamble_settings"]["end"]["threshold"]) in (float, int):
            self.end_pre_threshold = params["preamble_settings"]["end"]["threshold"]
        else:
            print(f"Invalid type, using default parameter for [\"preamble_settings\"][\"end\"][\"threshold\"])")
            self.end_pre_threshold = self.default_settings["preamble_settings"]["end"]["threshold"]

        if type(params["ecc"]["rs_ratio"]) in (float, int):
            self.rs_ratio = params["ecc"]["rs_ratio"]
        else:
            print(f"Invalid type, using default parameter for [\"ecc\"][\"rs_ratio\"])")
            self.rs_ratio = self.default_settings["ecc"]["rs_ratio"]