"""
Senses: hand-built encoders that turn what is in the arena into firing rates of the fly's *real*
sensory neurons. Each module documents which neuPrint cell types it drives and why.

* :mod:`vision`      a compound-eye retina model driving columnar motion detectors (T4/T5) and,
                     where the wiring does not carry the signal, the feature detectors (LC4, LPLC2,
                     LC10a) computed from the retinal image
* :mod:`olfaction`   odour plumes -> olfactory receptor neurons of specific glomeruli
* :mod:`taste`       food under the mouthparts and pheromones under the forelegs -> gustatory neurons
* :mod:`mechano`     wind, touch, dust and sound -> Johnston's organ and bristle neurons
"""
