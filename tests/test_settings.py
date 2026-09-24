"""Named model profiles."""

import pytest

from virtual_fly.plasticity import MushroomBodyPlasticity
from virtual_fly.settings import PROFILES, Profile, build_brain


def test_profiles_are_documented():
    assert set(PROFILES) == {"pure", "game", "brakes"}
    for name, p in PROFILES.items():
        assert isinstance(p, Profile) and p.name == name and p.description


def test_pure_profile_is_the_paper_model(conn):
    b = build_brain(conn, "pure")
    assert b.plasticity is None and b.silenced == {} and b.modulated == {}
    assert b.fatigue_mv == 0 and b.std_u == 0 and b.noise_hz == 0 and b.threshold_jitter == 0
    assert b.kenyon_gain == 0.25 and b.gain == 0.65 and b.dt == 0.5
    assert b.settings()["plasticity"] is None


def test_game_profile_silences_and_attaches_plasticity(conn):
    b = build_brain(conn, "game")
    assert set(b.silenced) == {"class:ALLN", "class:DAN"}
    assert b.silenced_mask().sum() == conn.count("class:ALLN") + conn.count("class:DAN") == 20
    assert (b.w[conn.out_edges(conn.select("class:DAN"))] == 0).all()
    assert isinstance(b.plasticity, MushroomBodyPlasticity) and b.plasticity.brain is b
    assert b.plasticity.pe.size == conn.edges_between("class:Kenyon_Cell", "class:MBON").size
    assert b.fatigue_mv == 0.05 and b.kenyon_gain == 1.0 and b.std_u == 0 and b.noise_hz == 0
    assert b.settings()["silenced"] == ["class:ALLN", "class:DAN"] and b.settings()["plasticity"]["enabled"]


def test_brakes_profile_uses_short_term_depression(conn):
    b = build_brain(conn, "brakes")
    assert b.std_u == 0.1 and b.std_tau_ms == 100.0 and b.std_x is not None and b.fatigue_mv == 0.05
    assert set(b.silenced) == {"class:DAN"} and b.plasticity is not None and b.kenyon_gain == 1.0


def test_overrides_and_unknown_profile(conn):
    b = build_brain(conn, "game", seed=7, fatigue_mv=0.0, noise_hz=2.0, noise_mv=0.5)
    assert b.seed == 7 and b.fatigue_mv == 0.0 and b.noise_hz == 2.0 and b.noise_mv == 0.5
    assert set(b.silenced) == {"class:ALLN", "class:DAN"}                  # profile parts still applied
    with pytest.raises(ValueError, match="unknown profile"):
        build_brain(conn, "turbo")


def test_profile_build_fails_loudly_when_a_silenced_population_is_missing(conn):
    p = Profile("odd", "silences a type that does not exist", silence=("class:NoSuchClass",))
    with pytest.raises(ValueError, match="no neurons match"):
        p.build(conn)
