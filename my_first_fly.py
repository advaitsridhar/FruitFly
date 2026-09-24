"""
my_first_fly.py - the smallest possible fly-brain experiment. Change a line, run it again.

    python my_first_fly.py

Every experiment has the same three steps: poke some neurons, let time pass, listen to others.
"""
from virtual_fly import load_connectome, FlyBrain

brain = FlyBrain(load_connectome())

# 1. Poke: make the sugar-sensing neurons on the fly's mouthparts fire 120 times per second
brain.stimulate("LB3b,LB3c", 120)

# 2. Wait: simulate half a second of fly time (all 176,422 neurons)
brain.run(500)

# 3. Listen: how fast is MN9, the motor neuron that pushes out the proboscis, firing?
print("MN9 (proboscis) fires at", round(brain.rate("MN9")), "spikes per second")

# Things to try next (each one is a real neuroscience question):
#
#  * Add bitter taste as well:   brain.stimulate("LB1a,LB1b,LB1c,LB1d", 120)
#    Does the fly still want to eat?
#
#  * Knock out a relay neuron before running, then compare MN9:
#        brain.silence("GNG232")     # G2N-1
#        brain.silence("DNge080")    # "Rounddown"
#        brain.silence("DNg67")      # "Fudog"
#    Which ones does the sugar signal actually need?  (python fly_brain.py --lesion Sugar --readout MN9)
#
#  * Ask the wiring diagram directly how sugar reaches MN9:
#        from virtual_fly import trace
#        for path in trace(brain.conn, "LB3b,LB3c", "MN9", max_hops=4):
#            print(path.describe())
#
#  * Swap the whole experiment for an escape:
#        brain.stimulate("LC4/R,LPLC2/R", 150)   # looming detectors, right eye
#        ...and listen to "DNp01", the giant fibre that triggers the escape jump.
#
#  * Teach it something (the game's settings: python -c "from virtual_fly.settings import build_brain"):
#        from virtual_fly.settings import build_brain
#        brain = build_brain(load_connectome(), "game")     # fatigue, calm antennal lobe, plasticity on
#        brain.stimulate("ORN_DM1,ORN_DM4,ORN_VM7d,ORN_DP1m", 80)   # the smell of vinegar
#        brain.stimulate("prefix:PPL1", 80)                          # ... paired with punishment dopamine
#        brain.run(2000)
#        print(brain.plasticity.summary()["MBON11"])                 # its synapses onto MBON11 are weaker now
#
#  * Look up neuron names:  python fly_brain.py --find DNa
#    Every name comes from neuPrint: https://neuprint.janelia.org (dataset male-cns:v1.0)
