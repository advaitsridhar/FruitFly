"""
Every population spec the kit's code names (the two-flies work, docs/TWO_FLIES_PLAN.md 9.5): the experiments' stimuli and
readouts, the game's readouts, zap presets and decoder targets, the senses' constants, the parts list's cells. Two users:
``tools/alias_audit.py`` counts each spec's cells on every connectome file (a union with one empty member must not hide
behind the others), and the BANC builder (``banc.py``) derives aliases for the MaleCNS type names BANC spells differently.
The imports are lazy: this module is cheap to import, the collection loads the game's modules.
"""
from __future__ import annotations

import re

_TERM_PREFIXES = ("prefix:", "class:", "superclass:", "subclass:", "nt:", "nerve:", "neuromere:", "gene:", "body:", "regex:",
                  "fbbt:", "dimorphism:", "frudsx:", "!")


def kit_specs() -> dict[str, str]:
    """{where: spec} for every spec the kit's code names, in a stable order."""
    out: dict[str, str] = {}

    def add(where: str, spec) -> None:
        if isinstance(spec, str) and spec.strip() and spec not in ("all",):
            out.setdefault(f"{where}", spec.strip())

    from . import experiments, game
    for exp in experiments.all_experiments():
        for spec in exp.stimulus:
            add(f"experiment '{exp.name}' stimulus", spec)
        for r in exp.readouts:
            add(f"experiment '{exp.name}' readout {r.label}", r.spec)
        for spec in exp.silence:
            add(f"experiment '{exp.name}' silenced", spec)
    for rows, what in ((game.READOUTS, "readout"), (getattr(game, "FEMALE_READOUTS", []), "female readout")):
        for row in rows:
            add(f"game {what} {row[0]}", row[1])
    for key, spec in game.HIDDEN_READOUTS.items():
        add(f"game hidden readout {key}", spec)
    for spec, _hz, label in game.ZAP_PRESETS:
        add(f"game zap {label[:40]}", spec)
    for name in ("COURTSHIP_SPEC", "SOUND_SPEC", "REWARD_SPEC", "SHOCK_SPEC", "BASELINE_SILENCED"):
        add(f"game {name}", getattr(game, name, None))
    for spec in getattr(game, "PHEROMONE_GRNS", {}):
        add("game PHEROMONE_GRNS", spec)
    decoder_specs = ("DNp09", "DNg100", "DNge053", "DNge050", "DNg97", "DNa02", "DNg13", "DNa01", "MDN", "DNg60",
                     "DNg74_a,DNg74_b", "AN19A018", "DNp01", "DNg62", "DNge078", "pIP10", "MN9")
    for spec in decoder_specs:                                   # game.MotorDecoder._measure_targets
        add(f"decoder {spec}", spec)
    from .senses import mechano, social
    for mod, names in ((mechano, ("WIND_LEFT", "WIND_RIGHT", "SOUND", "HEAD_BRISTLES", "LEG_PROPRIO")),
                       (social, ("SAG_SPEC",))):
        for name in names:
            add(f"senses {mod.__name__.rsplit('.', 1)[-1]} {name}", getattr(mod, name, None))
    try:
        from .senses import taste, olfaction
        for mod in (taste, olfaction):
            for name, val in vars(mod).items():
                if name.isupper() and isinstance(val, str) and _looks_like_spec(val):
                    add(f"senses {mod.__name__.rsplit('.', 1)[-1]} {name}", val)
                if name.isupper() and isinstance(val, dict) and mod is not olfaction:   # olfaction's dicts are keyed by odour
                    for k in val:
                        if isinstance(k, str) and _looks_like_spec(k):
                            add(f"senses {mod.__name__.rsplit('.', 1)[-1]} {name} key", k)
    except ImportError:
        pass
    try:
        from . import parts
        for name, val in vars(parts).items():
            if name.isupper() and isinstance(val, (list, tuple)):
                for item in val:
                    spec = item[0] if isinstance(item, (list, tuple)) and item else item
                    if isinstance(spec, str) and _looks_like_spec(spec):
                        add(f"parts {name}", spec)
    except ImportError:
        pass
    return out


def _looks_like_spec(text: str) -> bool:
    t = text.strip()
    if not t or " " in t.strip(",") and not t.startswith(_TERM_PREFIXES):
        return False
    return t.startswith(_TERM_PREFIXES) or bool(re.fullmatch(r"[A-Za-z0-9_'/\-,.!:*^$\[\]+]+", t))


def terms(spec: str) -> list[str]:
    """A spec's comma-joined terms, as Connectome.terms splits them (a plain type name, prefix:..., body:..., !x, ...)."""
    return [t.strip() for t in spec.split(",") if t.strip()]


def kit_alias_candidates() -> list[str]:
    """The plain type names plus the ``regex:`` terms the kit's specs use: what the BANC builder tries to alias."""
    found = set(kit_type_names())
    for spec in kit_specs().values():
        for t in terms(spec):
            if t.startswith("regex:"):
                found.add(t)
    return sorted(found)


def kit_type_names() -> list[str]:
    """The plain MaleCNS type names the kit's specs use (every term that is not a prefixed selector), sorted."""
    names = set()
    for spec in kit_specs().values():
        for t in terms(spec):
            if t.startswith(_TERM_PREFIXES) or t.startswith("&") or "&" in t:
                for part in t.split("&"):
                    part = part.strip().lstrip("!")
                    if part and not part.startswith(_TERM_PREFIXES):
                        names.add(part)
                continue
            names.add(t.lstrip("!"))
    return sorted(names)
