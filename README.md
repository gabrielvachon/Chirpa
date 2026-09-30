# Chirpa
A robust way to transfer small data via audio! Works by encoding and sending audio chirp signals into the air and interpreted by a decoder.

## Documentation
### Encoder
This is how you can encode a message:
```python
import json
import chirp_encoder

dyn = False
rs = True
file_content = open("comm_settings.json", "r")
settings = json.loads(file_content.read())

message = "Hello, World!"
print(f"Message lenght: {len(message)}")
enc = chirp_encoder.encoder(rs=rs, params=settings)
```

### Decoder
You can decode in realtime or decode an audio file.
#### Realtime decoding
Realtime decoder implementation example:
```python
import sounddevice as sd
import chirp_decoder
import json

file_content = open("comm_settings.json", "r")
settings = json.loads(file_content.read())

sd.default.device[0] = 1
channels = sd.query_devices(sd.default.device[0])["max_input_channels"]

dec = chirp_decoder.decoder(rs=True, graphical=False)

with sd.InputStream(
    samplerate=fs,
    blocksize=blocksize,
    channels=channels,
    dtype='float32'
) as stream:
    print("Recording. Ctrl+C to stop.")
    try:
        while True:
            data, overflowed = stream.read(blocksize)
            if overflowed:
                print("Overflow")

            block = data[:, 0]
            msg = dec.decode(block)

            if msg is not None:
                print(f"\nReceived: {msg}")

            if dec.state == "acq":
                prt = "Receiving..."
            elif dec.state == "decoding":
                prt = "Decoding..."
            else:
                prt = "Awaiting data..."

            print(f"\r{prt}".ljust(40), end='', flush=True)
    except KeyboardInterrupt:
        print("\nStopping.")
```

#### File decoding
File decoding implementation example:
```python
import json
from scipy.io import wavfile
from chirp_decoder import decoder

dyn = False
rs = True
file_content = open("comm_settings.json", "r")
settings = json.loads(file_content.read())

_, data = wavfile.read(filename="test123.wav")

dec = decoder(rs=rs, graphical=True, params=settings)
out = dec.decode(data)

print(out)

dec.show() # To display the correlation curves
```

### Import custom settings
If you want to customize the encoder/decoder settings, you may arrange them in the following format.
```json
{
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
```
#### Setting description
- `chirp_settings` : Settings for the data part audio signal
- `preamble_settings` : Settings for start/end preamble of the protocol
    - `freq_range` : Frequency range for both start AND end preambles
        - `f0` : Start frequency
        - `f1` : End frequency
    - `start/end`
        - `duration` : The duration of the preamble
        - `delay` : The time gap between data and the preamble
        - `threshold` : The minimum correlation value for the preamble to be
        - `window`
            - `type` : Currently works with Kaiser window only
            - `beta` : The beta factor for the Kaiser window 
- `decoder` : Related to the decoder only
    - `start_bit_range` : When the correlation data is received, it is split into its respective amount of symbols. The start range, between 0 and 1 (maximum of 0.5 recommended), is a ratio at which the decoder will start to decode a binary chirp. E.g. : If a data chirp of `N` samples is being decoded with a start range of 0.1, the decoder will start at sample `N*0.1`.
    - `end_bit_range` : The end range, between 0 and 1 (minimum of 0.5 recommended), is a ratio at which the decoder will stop the decoding of a binary chirp. E.g. : If a data chirp of `N` samples is being decoded with a end range of 0.9, the decoder will start at sample `N*0.9`.
    - `correlation_power_factor` : The amplification power factor of the decoded correlation signal.
    - `candidate_selection_mode` : Symbol comparition mode used to determine which bit value to assing considering the correlation signal candidates of `0` and `1`. `rms` (default) calculates the RMS value of both signal and pick the highest value between the candidate correlations. `max` takes the maximum correlation peak value between the correlation signals.  
    - `candidate_window` : Windowing used to lower the noise from the edges of specified range. Only works with Kaiser window.
    - `candidate_window_beta` : The Kaiser window beta factor for the candidate window.
- `ecc` : For Reed-Solomon ECC
    - `rs_ratio` : The codes to data ratio. E.g. : A raw message of lenght 30 will have 15 codes and thus will have a total lenght of 45 including ECC.