"""
A small synthetic connectome in the FLYB v1 binary format that ``virtual_fly.connectome.Connectome``
reads, so the test suite never needs the real 23 MB MaleCNS file.

The brain has ~600 neurons whose cell-type, class and superclass names match the ones the package
refers to, and a wiring diagram with *designed* pathways whose behaviour the tests check:

    sugar    LB3b/LB3c (and PhG1*, LgLG3) -> GNG232 -> MN9      (+ GNG232 -> DNg67 -> MN9)
    bitter   LB1a-d -> GNG087 (GABA) --| GNG232, MN9 ;  LB1a-d -> PPL101 (punishment dopamine)
    escape   LC4/x, LPLC2/x -> DNp01/x -> TTMn/x
    steering LC10a/x -> AOTU019/x -> DNa02/x   (weak alternative: LC10a/x -> AOTU025/x -> DNa02/x)
    motion   Mi1/Mi9/Mi4/C3 -> T4a-d and Tm1/Tm2/Tm9/Tm4 -> T5a-d per medulla column (the reference cells
             carry the hex column coordinates, as in the real data); T4a/T5a -> HSE/HSN/HSS -> DNp15
    sound    JO-A1/JO-B1 -> DNp01 (a loud sound reaches the giant fibre); JO-B1 -> OA-VPM3 (octopamine,
             onto the HS cells) and CSD (serotonin, onto the projection neurons): the parts list's modulators
    smell    ORN_<glom> -> <glom>_PN -> Kenyon cells -> MBONs ; ALLN and APL inhibition
    learning PPL101 -> MBON11, MBON12 ; PAM01 -> MBON01, MBON02 (DAN->MBON defines the compartment)
    touch    BM_InOm -> GNG_mdn -> MDN, BM_InOm -> MDN
    dust     JO-CA1, JO-EV1, BM_Ant -> WED -> DNg62, DNge078
    courting LgLG4 -> vAB3 -> pC1_1a -> pIP10 -> wing motor neurons
    walking  descending neurons -> premotor interneurons -> leg motor neurons (T1-T3)

Twenty neurons are unannotated (type ""), have no soma and no connections, so that the code paths
that skip them are exercised.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np

DATASET = "synthetic-fly:test"

TYPES_HEAD = [""]                 # index 0 = unannotated (the loader treats type_idx 0 as "no type")
SUPERCLASSES = ["", "sensory", "cb_intrinsic", "cb_motor", "descending_neuron", "vnc_motor",
                "vnc_intrinsic", "vnc_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
                "ascending_neuron"]
CLASSES = ["", "gustatory", "olfactory", "Kenyon_Cell", "MBON", "DAN", "ALLN", "ALPN", "APL",
           "mechanosensory", "mechanosensory_proprioceptive", "interneuron", "motor", "descending",
           "visual", "courtship"]
SUBCLASSES = ["", "sugar", "bitter", "water", "wind_gravity", "grooming", "gamma", "alpha_beta", "auditory", "leg"]
NTS = ["", "acetylcholine", "gaba", "glutamate", "dopamine", "unclear", "octopamine", "serotonin"]
SIGN_OF_NT = {"": 1, "acetylcholine": 1, "gaba": -1, "glutamate": -1, "dopamine": 1, "unclear": 1,
              "octopamine": 1, "serotonin": 1}
SIDES = ["", "L", "R", "M"]
DIMORPHISMS = ["", "isomorphic", "sexually dimorphic", "potentially sexually dimorphic", "male-specific", "potentially male-specific"]
FRUDSX = ["", "fru_high", "fru_low", "dsx_high", "dsx_low", "coexpress_high", "coexpress_low"]   # the MaleCNS labels
NEUROMERES = ["", "CB", "GNG", "T1", "T2", "T3"]
NERVES = ["", "ADMN", "LN", "PrN", "AN"]

# the olfactory glomeruli the Nose can drive (vinegar + banana + geosmin's DA2)
GLOMERULI = ["DM1", "DM4", "VM7d", "DP1m", "VA2", "DM2", "DM3", "VM2", "DC2", "VA6", "DA2"]
VINEGAR = GLOMERULI[:5]
BANANA = GLOMERULI[5:10]
HEX_LATTICE = [(h1, h2) for h1 in range(3) for h2 in range(2)]   # 6 medulla columns per eye


class _Builder:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.rows: list[dict] = []
        self.edges: dict[tuple[int, int], int] = {}
        self.groups: dict[str, list[int]] = {}          # "TYPE/side" -> neuron indices
        self.types: list[str] = list(TYPES_HEAD)

    # -------------------------------------------------------------- neurons
    def add(self, type_: str, n: int, side: str, superclass: str, cls: str, nt: str, subclass: str = "",
            neuromere: str = "", nerve: str = "", soma=None, hexes=None, dimorphism: str = "isomorphic",
            frudsx: str = "") -> list[int]:
        if type_ and type_ not in self.types:
            self.types.append(type_)
        idx = []
        for k in range(n):
            i = len(self.rows)
            hx = hexes[k] if hexes is not None else (-1, -1)
            if soma is None:
                pos = (float("nan"),) * 3
            else:
                cx, cy, cz = soma
                pos = (cx + self.rng.normal(0, 3000), cy + self.rng.normal(0, 3000), cz + self.rng.normal(0, 3000))
            self.rows.append(dict(type=type_, side=side, superclass=superclass, cls=cls, nt=nt, subclass=subclass,
                                  neuromere=neuromere, nerve=nerve, soma=pos, hex=hx, dimorphism=dimorphism,
                                  frudsx=frudsx))
            idx.append(i)
        key = f"{type_}/{side}" if side else type_
        self.groups.setdefault(key, []).extend(idx)
        return idx

    def both(self, type_: str, n: int, *args, soma=None, **kw) -> dict[str, list[int]]:
        """Add ``n`` neurons of a type on each side; left somata get positive x."""
        out = {}
        for side, sx in (("L", 1.0), ("R", -1.0)):
            s = None if soma is None else (soma[0] * sx, soma[1], soma[2])
            out[side] = self.add(type_, n, side, *args, soma=s, **kw)
        return out

    # -------------------------------------------------------------- edges
    def connect(self, pre: list[int], post: list[int], n_syn: int, jitter: int = 0):
        for a in pre:
            for b in post:
                if a == b:
                    continue
                s = n_syn + (int(self.rng.integers(-jitter, jitter + 1)) if jitter else 0)
                self.edges[(a, b)] = self.edges.get((a, b), 0) + max(5, s)

    # -------------------------------------------------------------- output
    def write(self, path: Path, n_retina: int = 0):
        n = len(self.rows)
        by_pre: dict[int, list[tuple[int, int]]] = {}
        for (a, b), s in self.edges.items():
            by_pre.setdefault(a, []).append((b, min(s, 65535)))
        row_ptr = [0]
        post_idx, n_syn = [], []
        for i in range(n):
            outs = sorted(by_pre.get(i, []))
            post_idx.extend(b for b, _ in outs)
            n_syn.extend(s for _, s in outs)
            row_ptr.append(len(post_idx))
        post_idx = np.asarray(post_idx, dtype="<i4")
        n_syn = np.asarray(n_syn, dtype="<u2")
        row_ptr = np.asarray(row_ptr, dtype="<i4")
        n_pre = np.zeros(n, dtype="<i4")          # synapses made (outgoing)
        n_post = np.zeros(n, dtype="<i4")         # synapses received
        np.add.at(n_pre, np.repeat(np.arange(n), np.diff(row_ptr)), n_syn.astype(np.int64))
        np.add.at(n_post, post_idx, n_syn.astype(np.int64))

        def tab(name, values):
            return np.asarray([name.index(v) for v in values])

        types = tab(self.types, [r["type"] for r in self.rows]).astype("<i4")
        body_id = (100000 + 7 * np.arange(n)).astype("<i8")
        parts = [b"FLYB", struct.pack("<IIII", 1, n, len(post_idx), n_retina)]

        def s16(s: str) -> bytes:
            b = s.encode("utf-8")
            return struct.pack("<H", len(b)) + b

        parts.append(s16(DATASET))
        meta = json.dumps({"source": "synthetic (virtual_fly test suite)", "min_synapses": 5,
                           "neurons": n, "connections": int(len(post_idx))}).encode()
        parts.append(struct.pack("<I", len(meta)) + meta)
        for table in (self.types, SUPERCLASSES, CLASSES, SUBCLASSES, NTS, SIDES, DIMORPHISMS, FRUDSX, NEUROMERES, NERVES):
            parts.append(struct.pack("<H", len(table)) + b"".join(s16(t) for t in table))
        soma = np.asarray([r["soma"] for r in self.rows], dtype="<f4").reshape(-1)
        arrays = [
            body_id, types,
            tab(SUPERCLASSES, [r["superclass"] for r in self.rows]).astype("u1"),
            tab(CLASSES, [r["cls"] for r in self.rows]).astype("u1"),
            tab(SUBCLASSES, [r["subclass"] for r in self.rows]).astype("<u2"),
            tab(NTS, [r["nt"] for r in self.rows]).astype("u1"),
            np.asarray([SIGN_OF_NT[r["nt"]] for r in self.rows], dtype="i1"),
            tab(SIDES, [r["side"] for r in self.rows]).astype("u1"),
            np.asarray([r["hex"][0] for r in self.rows], dtype="i1"),
            np.asarray([r["hex"][1] for r in self.rows], dtype="i1"),
            tab(DIMORPHISMS, [r["dimorphism"] for r in self.rows]).astype("u1"),
            tab(FRUDSX, [r["frudsx"] for r in self.rows]).astype("u1"),
            tab(NEUROMERES, [r["neuromere"] for r in self.rows]).astype("u1"),
            tab(NERVES, [r["nerve"] for r in self.rows]).astype("u1"),
            soma, n_pre, n_post, row_ptr, post_idx, n_syn,
        ]
        parts.extend(a.tobytes() for a in arrays)
        blob = b"".join(parts)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            with gzip.open(path, "wb") as f:
                f.write(blob)
        else:
            path.write_bytes(blob)
        return path


def build_synthetic(path: Path | str, seed: int = 7) -> Path:
    """Write the synthetic FLYB file to ``path`` (gzipped when the suffix is ``.gz``)."""
    b = _Builder(seed)
    add, both, con = b.add, b.both, b.connect
    S = "sensory"
    ACH, GABA, GLU, DA = "acetylcholine", "gaba", "glutamate", "dopamine"

    # ---------------------------------------------------------------- taste -> feeding
    lb3b = both("LB3b", 5, S, "gustatory", ACH, subclass="sugar", nerve="LN")
    lb3c = both("LB3c", 5, S, "gustatory", ACH, subclass="sugar", nerve="LN")
    phg = {t: both(t, 2, S, "gustatory", ACH, subclass="sugar", nerve="PrN") for t in ("PhG1a", "PhG1b", "PhG1c")}
    lglg3 = both("LgLG3", 3, S, "gustatory", ACH, subclass="sugar", neuromere="T1", nerve="LN")
    lb1 = {t: both(t, 4, S, "gustatory", ACH, subclass="bitter", nerve="LN") for t in ("LB1a", "LB1b", "LB1c", "LB1d")}
    lb2 = {t: both(t, 2, S, "gustatory", ACH, subclass="water", nerve="LN") for t in ("LB2a", "LB2b", "LB2c", "LB2d")}
    gng232 = both("GNG232", 2, "cb_intrinsic", "interneuron", ACH, neuromere="GNG", soma=(60000, 250000, 120000))
    gng087 = both("GNG087", 2, "cb_intrinsic", "interneuron", GABA, neuromere="GNG", soma=(70000, 250000, 125000))
    dng67 = both("DNg67", 1, "descending_neuron", "descending", ACH, neuromere="GNG", soma=(50000, 260000, 130000))
    mn9 = both("MN9", 1, "cb_motor", "motor", ACH, neuromere="GNG", nerve="PrN", soma=(40000, 255000, 140000))
    gng_m = add("GNG_M1", 2, "M", "cb_intrinsic", "interneuron", GABA, neuromere="GNG", soma=(0, 250000, 120000))
    for side in "LR":
        sugar = lb3b[side] + lb3c[side] + sum((phg[t][side] for t in phg), []) + lglg3[side]
        for other in "LR":                                   # taste neurons reach the relays on both sides
            con(sugar, gng232[other], 10, jitter=2)
            con(sum((lb1[t][other] for t in lb1), []), gng087[side], 15, jitter=3)
        con(gng232[side], mn9["L"] + mn9["R"], 80)            # the sugar relay drives both MN9s
        con(gng232[side], dng67[side], 25)
        con(dng67[side], mn9[side], 30)
        con(gng087[side], gng232[side], 60)                  # bitter inhibits the sugar relay ...
        con(gng087[side], mn9[side], 60)                     # ... and the motor neuron
        con(gng087[side], gng_m, 8)
        con(gng_m, gng232[side], 6)

    # ---------------------------------------------------------------- vision -> escape / steering
    lc4 = both("LC4", 8, "visual_projection", "visual", ACH, soma=(250000, 200000, 100000))
    lplc2 = both("LPLC2", 8, "visual_projection", "visual", ACH, soma=(255000, 200000, 105000))
    dnp01 = both("DNp01", 1, "descending_neuron", "descending", ACH, soma=(30000, 230000, 150000))
    ttmn = both("TTMn", 1, "vnc_motor", "motor", ACH, neuromere="T2", soma=(20000, 300000, 400000))
    lc10a = both("LC10a", 10, "visual_projection", "visual", ACH, frudsx="fru_low", soma=(240000, 190000, 95000))
    lc11 = both("LC11", 4, "visual_projection", "visual", ACH, soma=(245000, 195000, 98000))
    aotu019 = both("AOTU019", 4, "cb_intrinsic", "interneuron", ACH, soma=(120000, 180000, 90000))
    aotu025 = both("AOTU025", 2, "cb_intrinsic", "interneuron", ACH, soma=(118000, 182000, 92000))
    dna02 = both("DNa02", 1, "descending_neuron", "descending", ACH, soma=(40000, 220000, 150000))
    dna01 = both("DNa01", 1, "descending_neuron", "descending", ACH, soma=(42000, 222000, 150000))
    dna03 = both("DNa03", 1, "descending_neuron", "descending", ACH, soma=(44000, 224000, 150000))
    dng13 = both("DNg13", 1, "descending_neuron", "descending", ACH, neuromere="GNG", soma=(46000, 250000, 150000))
    dnp15 = both("DNp15", 1, "descending_neuron", "descending", ACH, soma=(41000, 221000, 150000))
    both("DNp51,DNpe019", 1, "descending_neuron", "descending", ACH, soma=(43000, 223000, 150000))   # a real name with a comma
    hs = {t: both(t, 1, "visual_centrifugal", "visual", ACH, soma=(230000, 200000, 110000)) for t in ("HSE", "HSN", "HSS")}
    t45 = {}
    for cell in ("T4", "T5"):
        for sub in "abcd":                                   # T4/T5 carry no column coordinate themselves
            t45[cell + sub] = both(cell + sub, len(HEX_LATTICE), "ol_intrinsic", "visual", ACH,
                                   soma=(280000, 200000, 100000))
    refs = {}
    for t in ("Mi1", "Mi9", "Mi4", "C3", "Tm1", "Tm2", "Tm9", "Tm4"):
        refs[t] = both(t, len(HEX_LATTICE), "ol_intrinsic", "visual", GABA if t in ("Mi4", "C3") else ACH,
                       hexes=HEX_LATTICE, soma=(285000, 200000, 100000))
    for side in "LR":
        for k in range(len(HEX_LATTICE)):                    # each column's inputs feed its own T4/T5 cells
            t4 = [t45[f"T4{sub}"][side][k] for sub in "abcd"]
            t5 = [t45[f"T5{sub}"][side][k] for sub in "abcd"]
            for t in ("Mi1", "Mi9", "Mi4", "C3"):
                con([refs[t][side][k]], t4, 20)
            for t in ("Tm1", "Tm2", "Tm9", "Tm4"):
                con([refs[t][side][k]], t5, 20)
        con(sum((hs[t][side] for t in hs), []), dnp15[side], 100)
        con(lc4[side] + lplc2[side], dnp01[side], 12, jitter=2)
        con(dnp01[side], ttmn[side], 120)
        con(lc10a[side], aotu019[side], 30, jitter=3)
        con(aotu019[side], dna02[side], 50)
        con(lc10a[side], aotu025[side], 5)
        con(aotu025[side], dna02[side], 5)
        con(lc11[side], dna02[side], 5)
        con(t45["T4a"][side] + t45["T5a"][side], sum((hs[t][side] for t in hs), []), 25)
        con(t45["T4b"][side] + t45["T5b"][side], hs["HSE"][side], 8)

    # ---------------------------------------------------------------- smell -> mushroom body
    orn, pn = {}, {}
    for g in GLOMERULI:
        orn[g] = both(f"ORN_{g}", 3, S, "olfactory", ACH, nerve="AN")
        pn_name = "DM1_lPN" if g == "DM1" else f"{g}_adPN"
        pn[g] = both(pn_name, 1, "cb_intrinsic", "ALPN", ACH, soma=(90000, 150000, 80000))
    alln = both("lLN1_a", 4, "cb_intrinsic", "ALLN", GABA, soma=(95000, 155000, 82000))
    kcg = both("KCg-m", 20, "cb_intrinsic", "Kenyon_Cell", ACH, subclass="gamma", soma=(130000, 120000, 60000))
    kcab = both("KCab-m", 10, "cb_intrinsic", "Kenyon_Cell", ACH, subclass="alpha_beta", soma=(132000, 118000, 62000))
    apl = both("APL", 1, "cb_intrinsic", "APL", GABA, soma=(135000, 125000, 65000))
    mbon = {"MBON11": both("MBON11", 1, "cb_intrinsic", "MBON", GABA, soma=(140000, 130000, 70000)),
            "MBON12": both("MBON12", 1, "cb_intrinsic", "MBON", ACH, soma=(141000, 131000, 70000)),
            "MBON01": both("MBON01", 1, "cb_intrinsic", "MBON", GLU, soma=(142000, 132000, 70000)),
            "MBON02": both("MBON02", 1, "cb_intrinsic", "MBON", GLU, soma=(143000, 133000, 70000)),
            "MBON20": both("MBON20", 1, "cb_intrinsic", "MBON", "unclear", soma=(144000, 134000, 70000))}
    ppl101 = both("PPL101", 2, "cb_intrinsic", "DAN", DA, soma=(150000, 140000, 75000))
    ppl102 = both("PPL102", 1, "cb_intrinsic", "DAN", DA, soma=(151000, 141000, 75000))
    pam01 = both("PAM01", 3, "cb_intrinsic", "DAN", DA, soma=(152000, 142000, 75000))
    for side in "LR":
        kcs = kcg[side] + kcab[side]
        for g in GLOMERULI:
            con(orn[g][side], pn[g][side], 60, jitter=5)
            con(orn[g][side], alln[side], 10)
        con(alln[side], sum((pn[g][side] for g in GLOMERULI), []), 5)
        # odour-specific Kenyon cells: 8 for vinegar, 8 for banana, the rest wired at random
        for k, kc in enumerate(kcs):
            if k < 8:
                gloms = list(b.rng.choice(VINEGAR, 4, replace=False))
            elif k < 16:
                gloms = list(b.rng.choice(BANANA, 4, replace=False))
            else:
                gloms = list(b.rng.choice(GLOMERULI, 3, replace=False))
            for g in gloms:
                con(pn[g][side], [kc], 40, jitter=5)
        con(kcs, apl[side], 5)
        con(apl[side], kcs, 5)
        for name in ("MBON11", "MBON12", "MBON01", "MBON02"):
            con(kcs, mbon[name][side], 15, jitter=3)
        con(ppl101[side], mbon["MBON11"][side] + mbon["MBON12"][side], 20)
        con(ppl102[side], mbon["MBON12"][side], 10)
        con(pam01[side], mbon["MBON01"][side] + mbon["MBON02"][side], 20)
        con(sum((lb1[t][side] for t in lb1), []), ppl101[side], 8)     # bitter -> punishment dopamine

    # ---------------------------------------------------------------- courtship
    lglg4 = both("LgLG4", 6, S, "gustatory", ACH, subclass="leg", neuromere="T1", nerve="LN")
    lglg1a = both("LgLG1a", 2, S, "gustatory", ACH, subclass="leg", neuromere="T1", nerve="LN")
    lglg1b = both("LgLG1b", 2, S, "gustatory", ACH, subclass="leg", neuromere="T1", nerve="LN")
    vab3 = both("vAB3", 2, "ascending_neuron", "interneuron", ACH, neuromere="T1", soma=(30000, 320000, 380000))
    pc1_1a = both("pC1_1a", 4, "cb_intrinsic", "courtship", ACH, dimorphism="male-specific", frudsx="coexpress_high",
                  soma=(100000, 170000, 85000))
    pc1_2a = both("pC1_2a", 1, "cb_intrinsic", "courtship", ACH, dimorphism="sexually dimorphic", frudsx="dsx_high",
                  soma=(102000, 172000, 85000))
    pip10 = both("pIP10", 1, "descending_neuron", "descending", ACH, frudsx="fru_high", dimorphism="male-specific",
                 soma=(45000, 230000, 150000))
    wing_mn = both("MNwm35", 2, "vnc_motor", "motor", ACH, neuromere="T2", soma=(15000, 310000, 410000))
    jo_a = both("JO-A1", 6, S, "mechanosensory", ACH, subclass="auditory", nerve="AN")
    jo_b = both("JO-B1", 6, S, "mechanosensory", ACH, subclass="auditory", nerve="AN")
    for side in "LR":
        con(lglg4[side] + lglg1a[side] + lglg1b[side], vab3[side], 30, jitter=4)
        con(vab3[side], pc1_1a[side] + pc1_1a["L" if side == "R" else "R"], 60)
        con(pc1_1a[side], pc1_2a[side], 20)
        con(pc1_1a[side] + pc1_2a[side], pip10[side], 90)
        con(pip10[side], wing_mn[side], 30)
        con(jo_a[side] + jo_b[side], pc1_1a[side], 5)
        con(jo_b[side], dnp01[side], 30)                     # a loud sound startles: JO-B -> giant fibre
        con(jo_a[side], dnp01[side], 5)

    # ---------------------------------------------------------------- neuromodulators (the parts list, parts.py)
    # an octopaminergic neuron onto the HS cells and one T4a column (Suver et al. 2012), and a serotonergic
    # CSD-like neuron onto the projection neurons; both are driven by the sound neurons so that a
    # stimulus can raise their tone
    oa = both("OA-VPM3", 1, "cb_intrinsic", "", "octopamine", soma=(160000, 190000, 100000))
    csd = both("CSD", 1, "cb_intrinsic", "", "serotonin", soma=(92000, 152000, 80000))
    for side in "LR":
        con(oa[side], sum((hs[t][side] for t in hs), []), 25)
        con(oa[side], [t45["T4a"][side][0]], 5)
        con(csd[side], sum((pn[g][side] for g in GLOMERULI), []), 10)
        con(jo_b[side], oa[side] + csd[side], 20)

    # ---------------------------------------------------------------- wind, dust, touch
    jo_c = both("JO-CA1", 6, S, "mechanosensory", ACH, subclass="wind_gravity", nerve="AN")
    jo_e = both("JO-EV1", 6, S, "mechanosensory", ACH, subclass="wind_gravity", nerve="AN")
    bm_ant = both("BM_Ant", 4, S, "mechanosensory", ACH, subclass="grooming", nerve="AN")
    wed = both("WED001", 2, "cb_intrinsic", "interneuron", ACH, soma=(110000, 240000, 110000))
    dng62 = both("DNg62", 1, "descending_neuron", "descending", ACH, neuromere="GNG", soma=(48000, 252000, 150000))
    dnge078 = both("DNge078", 1, "descending_neuron", "descending", ACH, neuromere="GNG", soma=(49000, 253000, 150000))
    bm_inom = both("BM_InOm", 6, S, "mechanosensory", ACH, nerve="ADMN")
    gng_mdn = both("GNG_mdn", 2, "cb_intrinsic", "interneuron", ACH, neuromere="GNG", soma=(65000, 255000, 125000))
    mdn = both("MDN", 2, "descending_neuron", "descending", ACH, neuromere="GNG", soma=(52000, 258000, 150000))
    for side in "LR":
        con(jo_c[side] + jo_e[side] + bm_ant[side], wed[side], 20, jitter=3)
        con(wed[side], dng62[side] + dnge078[side], 80)
        con(bm_inom[side], gng_mdn[side], 20)
        con(gng_mdn[side], mdn["L"] + mdn["R"], 40)
        con(bm_inom[side], mdn["L"] + mdn["R"], 15)

    # ---------------------------------------------------------------- walking: DNs -> premotor -> leg motor neurons
    walk_dns = {t: both(t, 1, "descending_neuron", "descending", ACH, soma=(38000, 235000, 150000))
                for t in ("DNp09", "DNg100", "DNge053", "DNge050", "DNg97", "DNg60", "DNg74_a", "DNg74_b")}
    an19 = both("AN19A018", 1, "ascending_neuron", "interneuron", ACH, neuromere="T1", soma=(25000, 330000, 390000))
    premotor, leg_mn = {}, {}
    for nm in ("T1", "T2", "T3"):
        premotor[nm] = both(f"IN{nm[1]}3B", 2, "vnc_intrinsic", "interneuron", ACH, neuromere=nm,
                            soma=(20000, 300000 + 30000 * int(nm[1]), 400000))
        leg_mn[nm] = both(f"MNleg_{nm}", 4, "vnc_motor", "motor", ACH, neuromere=nm, nerve="LN",
                          soma=(18000, 302000 + 30000 * int(nm[1]), 405000))
    proprio = both("LegCO", 4, "vnc_sensory", "mechanosensory_proprioceptive", ACH, neuromere="T1", nerve="LN")
    for side in "LR":
        for nm in premotor:
            con(sum((walk_dns[t][side] for t in walk_dns), []), premotor[nm][side], 20, jitter=3)
            con(dna02[side] + dna01[side] + dna03[side] + dng13[side] + mdn[side], premotor[nm][side], 15)
            con(premotor[nm][side], leg_mn[nm][side], 20)
            con(an19[side], premotor[nm][side], 10)
        con(proprio[side], premotor["T1"][side], 8)
        con(dnp01[side], premotor["T2"][side], 10)

    # ---------------------------------------------------------------- unannotated filler (no soma, no wiring)
    add("", 20, "", "", "", "")
    n_retina = sum(len(v[s]) for v in refs.values() for s in "LR")
    return b.write(path, n_retina=n_retina)


if __name__ == "__main__":                       # python tests/synthetic_connectome.py out.flyb.gz
    import sys
    out = build_synthetic(sys.argv[1] if len(sys.argv) > 1 else "synthetic.flyb.gz")
    print(out)


# ----------------------------------------------------------------------------------------------
# A hand-made slice of the anatomy ontology and of the receptor table for the synthetic types
# ----------------------------------------------------------------------------------------------

def build_mini_vfb():
    """``(Ontology, Receptors)`` built from dicts (no files): a few real FBbt ids and labels plus made-up
    classes for the test-only disagreements (a made-up class is marked "test" in its label)."""
    from virtual_fly.vfb import Ontology, Receptors
    C = {}

    def cls(cid, label, parents=(), symbol=None, definition=None):
        C[cid] = {"label": label, "parents": list(parents)}
        if symbol:
            C[cid]["symbol"] = symbol
        if definition:
            C[cid]["def"] = definition
    cls("FBbt:00005106", "neuron")
    cls("FBbt:00047095", "adult neuron", ["FBbt:00005106"])
    for cid, label in (("FBbt:00007173", "cholinergic neuron"), ("FBbt:00007228", "GABAergic neuron"),
                       ("FBbt:00100291", "glutamatergic neuron"), ("FBbt:00005131", "dopaminergic neuron"),
                       ("FBbt:00007364", "octopaminergic neuron"), ("FBbt:00005133", "serotonergic neuron"),
                       ("FBbt:00004101", "peptidergic neuron"), ("FBbt:00047097", "primary neuron")):
        cls(cid, label, ["FBbt:00005106"])
    cls("FBbt:00048000", "sNPF neuron", ["FBbt:00004101"])
    cls("FBbt:00003870", "lobula columnar neuron", ["FBbt:00047095"])
    cls("FBbt:00003874", "lobula columnar neuron LC4", ["FBbt:00003870", "FBbt:00007173", "FBbt:00100291"], "LC4",
        "Lobula columnar neuron whose cell body is in the lateral cell body rind (test copy).")
    cls("FBbt:00111747", "lobula columnar neuron LC10a", ["FBbt:00003870", "FBbt:00007173"], "LC10a")
    cls("FBbt:00100001", "lobula columnar neuron LC11", ["FBbt:00003870"], "LC11")
    cls("FBbt:00047511", "adult descending neuron", ["FBbt:00047095"])
    cls("FBbt:00004020", "giant fiber neuron", ["FBbt:00047511", "FBbt:00047097"], "DNp01")
    cls("FBbt:00047573", "descending neuron of the anterior dorsal brain DNa02", ["FBbt:00047511", "FBbt:00007173"], "DNa02")
    cls("FBbt:00049825", "adult Kenyon cell", ["FBbt:00047095", "FBbt:00007173", "FBbt:00048000"], "KC")
    cls("FBbt:00111061", "gamma main Kenyon cell", ["FBbt:00049825"], "KCg-m")
    cls("FBbt:00100248", "alpha/beta Kenyon cell", ["FBbt:00049825"], "KCab")
    cls("FBbt:00100246", "mushroom body output neuron 11", ["FBbt:00047095", "FBbt:00007228"], "MBON11")
    cls("FBbt:00100243", "mushroom body pedunculus-medial lobe arborizing neuron 1", ["FBbt:00047095", "FBbt:00005131"], "PPL101")
    cls("FBbt:00111015", "dopaminergic PAM neuron 1 (test: also GABAergic)", ["FBbt:00047095", "FBbt:00005131", "FBbt:00007228"], "PAM01")
    cls("FBbt:00007405", "adult CSD interneuron", ["FBbt:00047095", "FBbt:00005133"], "CSD")
    cls("FBbt:00090001", "octopaminergic VPM3 neuron (test)", ["FBbt:00047095", "FBbt:00007364"], "OA-VPM3")
    cls("FBbt:00090002", "adult GNG087 neuron (test: cholinergic)", ["FBbt:00047095", "FBbt:00007173"], "GNG087")
    cls("FBbt:20090003", "adult MBON20 neuron (test: glutamatergic)", ["FBbt:00047095", "FBbt:00100291"], "MBON20")
    cls("FBbt:00090005", "adult LB1a neuron (test: octopaminergic)", ["FBbt:00047095", "FBbt:00007364"], "LB1a")
    cls("FBbt:00003919", "equatorial giant horizontal cell HSE", ["FBbt:00047095"], "HSE")
    cls("FBbt:00090006", "adult DNg74 neuron (test)", ["FBbt:00047511"], "DNg74")
    tree = {"source": {"ontology": "test slice", "release": "test", "licence": "CC-BY 4.0"},
            "roots": {"neuron": "FBbt:00005106", "adult": "FBbt:00047095", "peptidergic": "FBbt:00004101",
                      "primary": "FBbt:00047097", "secondary": "FBbt:00047096",
                      "transmitters": {"acetylcholine": "FBbt:00007173", "gaba": "FBbt:00007228", "glutamate": "FBbt:00100291",
                                       "dopamine": "FBbt:00005131", "octopamine": "FBbt:00007364", "serotonin": "FBbt:00005133"}},
            "classes": C}

    def ev(cid):
        return "connectome" if cid.startswith("FBbt:2") else "literature"
    types = {}
    for t, ids, route, nts in (("LC4", ["FBbt:00003874"], "obo_symbol", ["acetylcholine", "glutamate"]),
                               ("LC10a", ["FBbt:00111747"], "obo_symbol", ["acetylcholine"]),
                               ("LC11", ["FBbt:00100001"], "obo_symbol", None),
                               ("DNp01", ["FBbt:00004020"], "obo_symbol", None),
                               ("DNa02", ["FBbt:00047573"], "obo_symbol", ["acetylcholine"]),
                               ("KCg-m", ["FBbt:00111061"], "obo_symbol", ["acetylcholine"]),
                               ("KCab-m", ["FBbt:00100248"], "name_in_male-cns", ["acetylcholine"]),
                               ("MBON11", ["FBbt:00100246"], "obo_symbol", ["gaba"]),
                               ("PPL101", ["FBbt:00100243"], "obo_symbol", ["dopamine"]),
                               ("PAM01", ["FBbt:00111015"], "obo_symbol", ["dopamine", "gaba"]),
                               ("CSD", ["FBbt:00007405"], "obo_symbol", ["serotonin"]),
                               ("OA-VPM3", ["FBbt:00090001"], "obo_symbol", ["octopamine"]),
                               ("GNG087", ["FBbt:00090002"], "obo_symbol", ["acetylcholine"]),
                               ("MBON20", ["FBbt:20090003"], "obo_symbol", ["glutamate"]),
                               ("LB1a", ["FBbt:00090005"], "obo_symbol", ["octopamine"]),
                               ("HSE", ["FBbt:00003919"], "obo_symbol", None),
                               ("DNg74_a", ["FBbt:00090006"], "obo_stem", None), ("DNg74_b", ["FBbt:00090006"], "obo_stem", None)):
        e = {"fbbt": ids, "route": route, "n": 1}
        if route == "obo_stem":
            e["coarse"] = True
        if nts:
            e["nt"] = nts
            e["evidence"] = ev(ids[0])
        types[t] = e
    if True:
        types["KCab-m"]["coarse"] = True
    ont = Ontology({"source": tree["source"], "types": types}, tree)
    rx = Receptors({
        "source": {"families": {"FCA_MALE": {"label": "Fly Cell Atlas (male)"}, "DAVIE": {"label": "Davie 2018"}, "KURM": {"label": "Kurmangaliyev 2020"}}},
        "families": ["FCA_MALE", "DAVIE"],
        "genes": {"Dop1R1": {"fbgn": "FBgn0011582", "modulator": "dopamine", "sign": 1},
                  "Dop2R": {"fbgn": "FBgn0053517", "modulator": "dopamine", "sign": -1},
                  "Oamb": {"fbgn": "FBgn0024944", "modulator": "octopamine", "sign": 1},
                  "Octα2R": {"fbgn": "FBgn0038653", "modulator": "octopamine", "sign": -1},
                  "5-HT1A": {"fbgn": "FBgn0004168", "modulator": "serotonin", "sign": -1},
                  "5-HT7": {"fbgn": "FBgn0004573", "modulator": "serotonin", "sign": 1}},
        "clusters": {"FBlc1": {"family": "FCA_MALE", "stage": "adult"}, "FBlc2": {"family": "KURM", "stage": "pupal"},
                     "FBlc3": {"family": "DAVIE", "stage": "adult"}, "FBlc4": {"family": "FCA_MALE", "stage": "adult"}},
        "classes": {"FBbt:00111061": {"Dop1R1": [["FBlc1", 0.8, 2091.7], ["FBlc4", 0.8, 2091.7]], "Dop2R": [["FBlc1", 0.75, 2180.9], ["FBlc4", 0.75, 2180.9]],
                                      "5-HT1A": [["FBlc1", 0.3, 1209.5], ["FBlc4", 0.3, 1209.5]]},
                    "FBbt:00003919": {"Oamb": [["FBlc3", 0.3, 100.0]], "Octα2R": [["FBlc3", 0.6, 100.0]]},
                    "FBbt:00003870": {"Dop2R": [["FBlc3", 0.5, 1.0]]},
                    "FBbt:00047573": {"Dop1R1": [["FBlc2", 0.9, 1.0]]}}})
    return ont, rx
