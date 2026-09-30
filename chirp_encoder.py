import numpy as np
from chirp_utils import linear_chirp, exponential_chirp, hyperbolic_chirp, default_settings
from unireedsolomon import RSCoder

class encoder():
    # TODO : Try different encoding approaches, e.g.: use 3 state (up, down, none), use additional channels for encoding, etc.
    """# Chirpa decoder
    Used to encode byte messages using chirps.
    Note: Currently works only with *str* inputs.
    """
    def __init__(self, rs:bool, params:dict=default_settings):
        """
        Initializes an encoder. Creates bit models for each channel, preamble models, according to the specified settings.
        Args:
            **rs** : *bool*
                `True`, enables Reed-Solomon error correction codes
                `False`, Reed-Solomon is disabled.                
            **graphical** : *bool* (optional)
                `True`, allows processing of graphical data. Useful for debugging and visualizing. Computationally heavy.
                `False` (default), no graphical data will generated for this instance. Computationally lighter.
            **params** : *dict* (optional)
                Allows the user to input custom settings. The settings must be placed in a dictionary. The dictionary format must follow the documentation. Alternatively, you can follow the `default_settings` variable from `chirp_utils.py` to set the parameters.
        """
        self.default_settings = default_settings
        self.set_parameters(params)

        self.rs_enabled = rs

        self.N = int(self.fs * self.T)
        self.t = np.arange(self.N) / self.fs

        self.win = np.kaiser(M=self.N, beta=self.beta)

        # Start preamble settings
        self.N_start_pre = int(self.fs * self.start_pre_duration)
        self.t_start_pre = np.arange(self.N_start_pre) / self.fs
        self.start_pre_delay_lenght = int(self.fs * self.start_pre_delay)
        self.start_pre_lenght = int(self.fs * (self.start_pre_duration + self.start_pre_delay))
        self.start_pre_win = np.kaiser(self.N_start_pre, beta=self.start_pre_win_beta)

        # End preambule settings
        self.N_end_pre = int(self.fs * self.end_pre_duration)
        self.t_end_pre = np.arange(self.N_end_pre) / self.fs
        self.end_pre_delay_lenght = int(self.fs * self.end_pre_delay)
        self.end_pre_lenght = self.N_end_pre + self.end_pre_delay_lenght
        self.end_pre_win = np.kaiser(self.N_end_pre, beta=self.end_pre_win_beta)

    # TODO : Make other types than str compatible with this function (if possible)
    def encode(self, data):
        """
        Encodes a message into with the Chirpa audio codec using chirps.
        Args:
            **data** : *str*
                The message to encode. Currently works only for *str* type.
        Returns:
            **audio_data** : *numpy.ndarray*
                The encoded audio data buffer.
        """
        bit_range = 8
        if self.rs_enabled: 
            rs_enc = RSCoder(int(np.round(len(data) * self.rs_ratio)), len(data))
            data = rs_enc.encode(data)
        audio_data = np.zeros(self.start_pre_lenght + len(data) * self.N + self.end_pre_lenght)

        channel_arrangement_pre_start = range(bit_range)
        for n in channel_arrangement_pre_start:
            f_s_m = self.f0 + (self.f1 - self.f0) * n / bit_range
            f_e_m = f_s_m + (self.f1 - self.f0) / bit_range
            audio_data[:self.N_start_pre] += linear_chirp(self.t_start_pre, f0=f_s_m, f1=f_e_m, T=self.start_pre_duration) * self.start_pre_win
            audio_data[:self.N_start_pre] += linear_chirp(self.t_start_pre, f0=f_e_m, f1=f_s_m, T=self.start_pre_duration) * self.start_pre_win

        channel_arrangement_data = range(bit_range)
        for m in range(len(data)):
            for n in channel_arrangement_data:
                f_s_m = self.f2 + (self.f3 - self.f2) * n / bit_range
                f_e_m = f_s_m + (self.f3 - self.f2) / bit_range
                i = m * self.N
                data_range_start = self.start_pre_lenght + i
                data_range_end = self.start_pre_lenght + i + self.N
                data_content = audio_data[data_range_start : data_range_end]
                if (ord(data[m]) >> n) & 0b1:
                    data_content += linear_chirp(t=self.t, f0=f_s_m, f1=f_e_m, T=self.T) * self.win
                else:
                    data_content += linear_chirp(t=self.t, f0=f_e_m, f1=f_s_m, T=self.T) * self.win
                    
        channel_arrangement_pre_end = range(bit_range)
        max_channel_pre_end = channel_arrangement_pre_end.stop + 1
        for n in channel_arrangement_pre_end:
            f_s_m = self.f0 + (self.f1 - self.f0) * n / (max_channel_pre_end - 1)
            f_e_m = f_s_m + (self.f1 - self.f0) / (max_channel_pre_end - 1)
            end_start = self.start_pre_lenght + len(data) * self.N + self.end_pre_delay_lenght
            audio_data[end_start:end_start + self.N_end_pre] += linear_chirp(self.t_end_pre, f0=f_s_m, f1=f_e_m, T=self.end_pre_duration) * self.end_pre_win
            audio_data[end_start:end_start + self.N_end_pre] += linear_chirp(self.t_end_pre, f0=f_e_m, f1=f_s_m, T=self.end_pre_duration) * self.end_pre_win

        rms = np.sqrt(np.mean(audio_data ** 2))
        target_rms = 1
        audio_data = audio_data * (target_rms / rms)
        audio_data = np.clip(audio_data, -1, 1)

        return audio_data

    def set_parameters(self, params:dict):
        """
        Take a dictionary containing the settings as input. By default, default settings from the `default_settings` from `chirp_utils.py` will be used.
        """
        # GLOBAL SETTINGS
        if type(params["sample_rate"]) in (float, int):
            self.fs = params["sample_rate"]
        else:
            print(f"Invalid type, using default parameter for [\"sample_rate\"])")
            self.fs = self.default_settings["sample_rate"]

        # CHIRP SETTINGS
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

        #add window type or remove
        if type(params["chirp_settings"]["window"]["beta"]) in (float, int):
            self.beta = params["chirp_settings"]["window"]["beta"]
        else:
            print(f"Invalid type, using default parameter for [\"chirp_settings\"][\"window\"][\"beta\"])")
            self.beta = self.default_settings["chirp_settings"]["window"]["beta"]

        # GLOBAL PREAMBLE SETTINGS
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

        # START PREAMBLE SETTINGS
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

        # END PREAMBLE SETTINGS
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

        # REED-SOLOMON ERROR CORRECTION CODES
        if type(params["ecc"]["rs_ratio"]) in (float, int):
            self.rs_ratio = params["ecc"]["rs_ratio"]
        else:
            print(f"Invalid type, using default parameter for [\"ecc\"][\"rs_ratio\"])")
            self.rs_ratio = self.default_settings["ecc"]["rs_ratio"]
