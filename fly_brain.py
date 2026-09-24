#!/usr/bin/env python3
"""
fly_brain.py - run a whole fruit-fly nervous system on your own computer.

    pip install numpy
    python fly_brain.py                          # download the data (23 MB), run the validated experiments
    python fly_brain.py --find DNa               # search neuron types by name
    python fly_brain.py --stim MDN:60 --watch MDN,DNp09,DNa02
    python fly_brain.py --trace LC10a/L DNa02/L  # how does the eye reach the steering neurons?
    python fly_brain.py --help                   # everything else

This file is a thin entry point; the code lives in the ``virtual_fly`` package next to it.
Use it from your own code:

    from virtual_fly import load_connectome, FlyBrain
    brain = FlyBrain(load_connectome())
    brain.stimulate("LB3b,LB3c", 120)            # drive sugar-sensing neurons at 120 spikes/s
    brain.run(500)                               # simulate 500 ms
    print(brain.rate("MN9"))                     # proboscis motor neuron firing rate (Hz)
"""
from virtual_fly import FlyBrain, load_connectome, download_connectome, Connectome   # noqa: F401  (re-exported)
from virtual_fly.cli import main

if __name__ == "__main__":
    main()
