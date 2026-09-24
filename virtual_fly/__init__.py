"""
virtual_fly: a whole-nervous-system fruit fly you can play with.

    from virtual_fly import load_connectome, FlyBrain
    brain = FlyBrain(load_connectome())
    brain.stimulate("LB3b,LB3c", 120)
    brain.run(500)
    print(brain.rate("MN9"))

Modules: :mod:`connectome` (the data), :mod:`brain` (the simulation), :mod:`plasticity` (learning),
:mod:`pathways` (tracing routes through the wiring), :mod:`experiments` (validated protocols),
:mod:`senses` (retina, olfaction, mechanosensation), :mod:`body` and :mod:`world` (the fly and its
arena), :mod:`game` (the sensorimotor loop) and :mod:`server` (the browser front end).
"""

from .connectome import Connectome, DATA_FILE, download_connectome, load_connectome
from .brain import FlyBrain
from .plasticity import MushroomBodyPlasticity, mbon_valence
from .pathways import trace, relay_ranking
from . import experiments

__version__ = "2.0.0"
__all__ = ["Connectome", "DATA_FILE", "download_connectome", "load_connectome", "FlyBrain",
           "MushroomBodyPlasticity", "mbon_valence", "trace", "relay_ranking", "experiments", "__version__"]
