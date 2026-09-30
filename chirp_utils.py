import numpy as np

def linear_chirp(t, f0, f1, T):
    """# Linear chirp
    Generate a linear chirp.
    Args:
        **t** : *numpy.ndarray*
            Time array.
        **f0** : *int, float*
            Starting frequency of the chirp.
        **f1** : *int, float*
            Ending frequency of the chirp.
        **T** : *int, float*
            Period or duration of the chirp.
    Returns:
        **chirp** : *numpy.ndarray*
            The generated linear chirp signal.
    """
    return np.sin(2 * np.pi * (f0 * t +
        ((f1 - f0) * t ** 2) / (2 * T)))

def exponential_chirp(t, f0, f1, T):
    """# Exponential chirp
    Generate an exponential chirp.
    Args:
        **t** : *numpy.ndarray*
            Time array.
        **f0** : *int, float*
            Starting frequency of the chirp.
        **f1** : *int, float*
            Ending frequency of the chirp.
        **T** : *int, float*
            Period or duration of the chirp.
    Returns:
        **chirp** : *numpy.ndarray*
            The generated exponential chirp signal.
    """
    k = f1 / f0
    return np.sin(2 * np.pi * f0 *
        ((T * ((k ** (t / T)) - 1))
        / np.log(k)))

def hyperbolic_chirp(t, f0, f1, T):
    """# Linear chirp
    Generate an hyperbolic chirp.
    Args:
        **t** : *numpy.ndarray*
            Time array.
        **f0** : *int, float*
            Starting frequency of the chirp.
        **f1** : *int, float*
            Ending frequency of the chirp.
        **T** : *int, float*
            Period or duration of the chirp.
    Returns:
        **chirp** : *numpy.ndarray*
            The generated hyperbolic chirp signal.
    """
    return np.sin(-2 * np.pi *
        ((f0 * f1 * T) / (f1 - f0)) *
        np.log(1 - ((f1 - f0) / (f1 * T) * t)))

# Default settings use for Chirpa encoder and decoder.
# May be reproduced and modified in your own script or imported from external JSON file.
default_settings = {
    "sample_rate": 44.1e3,
    "blocksize": 8192,
    "filter_order": 4096,
    "chirp_settings":
    {
        "freq_range":
        {
            "f2": 10e3,
            "f3": 18e3
        },
        "chirp_period": 0.1,
        "type": "linear",
        "window":
        {
            "type": "kaiser",
            "beta": 8
        }
    },
    "preamble_settings":
    {
        "freq_range":
        {
            "f0": 7e3,
            "f1": 10e3
        },
        "start":
        {
            "duration": 0.75,
            "delay": 0.1,
            "window":{
                "type": "kaiser",
                "beta": 8
            },
            "threshold": 0.15
        },
        "end":
        {
            "duration": 0.25,
            "delay": 0.1,
            "window":{
                "type": "kaiser",
                "beta": 8
            },
            "threshold": 0.15
        }
    },
    "decoder":{
        "start_bit_range": 0.1,
        "end_bit_range": 0.9,
        "correlation_power_factor": 2,
        "candidate_selection_mode": "rms",
        "candidate_window": True,
        "candidate_window_beta": 24
    },
    "ecc":{
        "rs_ratio": 1.5
    }
}