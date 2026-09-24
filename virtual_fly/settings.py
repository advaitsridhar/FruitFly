"""
Named model profiles: which optional mechanisms are switched on, and why.

``pure``   the published model, nothing added. Bitter taste, dust and smells set off runaway
           loops in the antennal lobe that never stop (the model has none of the brakes real
           neurons have).
``game``   what the game runs. Two documented fixes for the runaway loops (mild spike-frequency
           adaptation, and blocking the output of the 420 antennal-lobe local neurons, many of
           which don't fire spikes in real flies anyway), dopamine neurons treated as slow
           modulators (their spikes gate plasticity, their fast synapses are muted), Kenyon-cell
           input gain raised so that the odour code is sparse *and* odour-specific once the
           antennal lobe is calm, and mushroom-body plasticity. Background noise is available
           (``--noise 2:1``) but off by default.
``brakes`` an experimental profile: short-term synaptic depression instead of silencing. It keeps
           the brain from seizing (every "after the stimulus" test is calm) but weakens strong
           sensory drive: only 6 of the 12 classic readouts stay in their published range (MN9 10 Hz,
           giant fibre 195 Hz, aDN1 32 Hz). Kept so you can see the trade-off yourself.

Every number here was checked against the classic experiments on this machine; see README.md
("What it does, and how it was checked") for the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .brain import FlyBrain
from .connectome import Connectome
from .plasticity import MushroomBodyPlasticity


@dataclass
class Profile:
    name: str
    description: str
    brain: dict = field(default_factory=dict)          # FlyBrain keyword arguments
    silence: tuple[str, ...] = ()                      # populations whose output is blocked
    plasticity: dict | None = None                     # MushroomBodyPlasticity kwargs, or None

    def build(self, conn: Connectome, **overrides) -> FlyBrain:
        kwargs = {**self.brain, **overrides}
        brain = FlyBrain(conn, **kwargs)
        for spec in self.silence:
            brain.silence(spec)
        if self.plasticity is not None:
            MushroomBodyPlasticity(**self.plasticity).attach(brain)
        return brain


PROFILES: dict[str, Profile] = {
    "pure": Profile("pure", "Shiu et al. 2024 exactly: no fatigue, nothing silenced, no noise, no learning."),
    "game": Profile(
        "game",
        "fatigue 0.05 mV/spike, antennal-lobe local neurons and dopamine outputs silenced, "
        "Kenyon gain 1.0, mushroom-body plasticity on (background noise off unless --noise)",
        brain=dict(fatigue_mv=0.05, kenyon_gain=1.0, noise_hz=0.0, noise_mv=1.0),
        # ALLN: the runaway fix. DAN: dopamine neurons act through slow receptors in real flies; here their
        # spikes gate plasticity but their (modelled-as-excitatory) fast synapses are muted so that they
        # do not drive the MBONs directly.
        silence=("class:ALLN", "class:DAN"),
        plasticity=dict(),
    ),
    "brakes": Profile(
        "brakes",
        "short-term synaptic depression (U 0.1, 100 ms) plus fatigue, only dopamine outputs silenced (experimental)",
        brain=dict(fatigue_mv=0.05, std_u=0.1, std_tau_ms=100.0, kenyon_gain=1.0),
        silence=("class:DAN",),
        plasticity=dict(),
    ),
}


def build_brain(conn: Connectome, profile: str = "game", **overrides) -> FlyBrain:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile '{profile}'; choose from {', '.join(PROFILES)}")
    return PROFILES[profile].build(conn, **overrides)
