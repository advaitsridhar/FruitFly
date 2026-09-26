# Game server API

`python fly_game.py` starts a local HTTP server (default `http://127.0.0.1:8765`). The browser
page is served from `virtual_fly/web/`. Everything the page does goes through this API, so any
program that speaks HTTP can drive the fly. `--host 0.0.0.0` makes it reachable from other computers
on your network: the API has no password and answers any origin (`Access-Control-Allow-Origin: *`).

## Static data

### `GET /api/layout`

Returned once at start-up (gzip-compressed if the client accepts it; ~2.4 MB raw).

| field | meaning |
|---|---|
| `n` | number of neurons (176,422) |
| `w`, `h`, `d` | extent of the brain map in map units |
| `x[]`, `y[]`, `z[]` | per-neuron map coordinates (ints, 0..~1000; `-1` = no soma known). `x` = left-right (fly's left on the left), `y` = top-bottom (head at top, nerve cord below), `z` = depth (anterior-posterior) |
| `region[]` | per-neuron region index into `regions` |
| `regions[]` | `["optic lobes","central brain","descending","nerve cord","ascending","motor","other","Kenyon cells","MBON / DAN"]` |
| `arena_r`, `fly_half`, `tick_ms` | arena radius (mm), fly half-length (mm), brain ms per tick (25) |
| `presets[]` | `{spec, hz, label}` zap presets for the neuron lab |
| `types[]` | cell type names (most numerous first) for autocomplete |
| `edges`, `synapses` | connection and synapse counts |
| `dataset`, `sex` | which connectome: `male-cns:v1.0` / `male`, or `flywire:v783` / `female` (`fly_game.py --female`) |
| `readouts[]` | `{key, spec, label, group, max, colour}`: the key-neuron bars, in display order, grouped by `group` (only those whose cells exist in this connectome: the female fly has no `pIP10` or `TTMn`) |
| `checks[]` | `{id, text}`: the experiments checklist |
| `odours[]` | `{id, name, glomeruli[], innate, colour, note}` |
| `scenarios[]` | `{id, name, description}` |
| `retina` | `{L: {n_az, n_el, az[], el[]}, R: {...}}` facet directions (radians, fly frame; `az` + = left) |
| `profile` | model profile name (`game`, `pure`, `brakes`) |
| `settings` | brain settings dict (dt, backend `numpy`/`numba`, gain, fatigue, silenced, plasticity ...) |
| `genetics` | `{expression: [{key, label, spec, n, high, types, gene, flybase}], transmitters: [{nt, spec, n, sign, synapse_share, genes: [{symbol, flybase}]}], unclear, unclear_inhibitory, genes: [...], readouts: {key: {n, fru, dsx, male, female, dimorphic, nt, tags}}, source}`; each `readouts[]` row also carries `genes` (its tags: `fru`, `dsx`, `♂`, `♀`, `♂♀`); `high` is null for the female fly, whose labels have no confidence grade; `unclear_inhibitory` counts the `unclear` neurons that inhibit (0 in MaleCNS, which counts them excitatory; FlyWire keeps the low-confidence prediction's sign) |
| `genome` | `{levels: [{level, label}]}`: the wiring levels the `grow` action accepts |
| `parts` | `{tables: {modulators: [{nt, label, genes[], receptors, tau_ms, gain, why}], graded: [{spec, label, why}], params: [...], graded_rate_hz, curated, receptor_signs, unknown_sign, receptors: [{gene, fbgn, modulator, coupling, sign, why}], receptor_facts: [{spec, receptors[], effects: {<modulator>: sign}, what, label, why}], local: [{spec, groups[], label, tau_ms, by_region, why}]}, counts: {modulators: [{nt, neurons, synapses, targets, co_release, ...}], modulatory_neurons, modulated_targets, co_release_neurons, graded: [{spec, label, neurons, why}], graded_neurons, params, local: [{spec, label, neurons, groups: [{spec, neurons}], mode, compartments, by_label: {<compartment>: {input_synapses, output_synapses}}, outputs, placed, tau_ms, why}], curated: {policy, types, neurons, by_action, signs_changed, confident_signs_flipped, rows: [{type, n, predicted, curated[], evidence, action, fbbt, label, source?}]}, receptor_signs: {on, coverage: [{nt, targets, with_data, mean_sign, negative}], facts: [{spec, label, receptors[], effects, what, neurons, targets, signs, left_to_the_atlas[], why}], receptors[]}}}`: the parts list (`parts.py`) and what it finds in this connectome, including what the curated transmitters change and how many modulated targets have receptor data (for the female fly, rows from FlyWire's own literature column carry `source` and may have an empty `fbbt`) |
| `vfb` | `{available, source, overlay_source, types_mapped, types_total, neurons_mapped, neurons_typed, classes, curated: {agree, differ, unclear_with_curated, differ_rows: [{type, n, predicted, curated[], evidence, fbbt, label, source?}]}}` (or `{available: false}` without the data files): the anatomy-ontology join (`vfb.py`); the counts cover only this fly's types, and for the female fly they include FlyWire's own literature column (known_nt): those rows carry `source`, and one whose type has no FBbt class has an empty `fbbt` and the type name as `label` |
| `decoder` | per decoder DN spec: motor synapses it reaches (`direct_motor_synapses`, `two_hop_motor_synapses_by_neuromere`) |
| `columnar_vision` | bool: T4/T5 columns driven from the retina |
| `body`, `stride_average` | which body walks: `drawn` (default) or `physics` (`--body physics`), and whether the senses see its stride-averaged pose (`--stride-average`) |
| `whats_real` | `{wiring[], hand_built[], not_modelled[]}` text for the "What's real here?" dialog |

## Live state

### `GET /api/stream` (Server-Sent Events) and `GET /api/state`

One JSON object per tick (40 per second at real time). Same schema on both endpoints.

| field | meaning |
|---|---|
| `seq` | increases every tick |
| `t` | fly time (s) |
| `rtf` | real-time factor the brain is achieving (1 = real time) |
| `speed` | requested time scale |
| `paused`, `autopilot` | bools |
| `mode` | `idle` (resting: no command moves the legs), `walk`, `feed`, `groom`, `escape`, `backward`, `court` |
| `fly` | `{x, y, h, v, w, mode, prob, legs, groom, wingL, wingR, abdomen, jump, hx, hy, dist}`: position mm, heading rad (0 = +x, CCW), forward speed mm/s, yaw rate rad/s, proboscis 0..1, gait phase, groom phase, wing extensions 0..1, abdomen bend 0..1, jump progress 0..1 or null, head position, distance walked; with `--body physics` also `physics: {left, right, z, tarsi[6][x, y]}` (the stepping drive per side, thorax height, tarsus positions) |
| `world` | `{food[], obstacles[], odours[], puffs[], wind, female, hand, tool}`: see below |
| `senses` | which senses are active now: keys `taste_sugar`, `taste_bitter`, `taste_water`, `small` (`"L"`,`"R"`,`"LR"`), `loom`, `flow`, `smell` (odour id), `pheromone`, `courting`, `sound`, `wind` (bearing in degrees the wind comes from, + = left; `0` means straight ahead, so test for the key, not the value), `dust`, `touch`, `reward`, `shock`, `zap` (text) |
| `retina` | `{L: base64, R: base64}`: one byte per facet (0 dark .. 255 bright), facet order matches `layout.retina` |
| `hz` | firing rate (Hz per neuron, smoothed) per readout key, including custom watches (a readout whose cells this fly lacks, such as the female's `pIP10`, is left out) |
| `motor` | decoder drives: `forward, yaw, backward, halt, feed, groom, song, court` (0..1; yaw -1..1, + = right) |
| `driver` | text explaining what drives the current behaviour |
| `spikes[]` | up to 2,500 neuron indices that spiked this tick (for the brain map) |
| `sps` | events per second in the whole brain: spikes, plus the graded cells' release quanta when the parts list is on (each quantum a spike's worth of transmitter) |
| `graded_eps` | of `sps`, the graded cells' release quanta per second (0 with the parts list off) |
| `stims` | number of stimulated populations |
| `calms` | how often the runaway watchdog reset the brain |
| `msg` | a toast message or `""` |
| `silenced[]`, `baseline[]`, `modulated{}` | user-silenced specs, profile-silenced specs, `{spec: factor}` |
| `custom{}` | custom watch readouts `{key: spec}` |
| `done[]` | ids of completed checklist items |
| `state` | `{hunger, thirst, arousal}` 0..1 |
| `learning` | `null` or `{enabled, depressed_fraction, events, learned_bias, smelling, mbon: {type: {strength, now, valence, dopamine}}}` (`strength` = mean remaining KC→MBON strength 0..1 over all Kenyon cells; `now` = the same weighted by the Kenyon cells active right now, i.e. for the odour being smelled; `valence` +1 approach / -1 avoid) |
| `events[]` | the last 12 events `{id, t, kind, text}`; `event_seq` is the newest id |
| `scenario` | `null` or `{id, name, step, steps, caption, left, measure{}}` |
| `recording` | `null` or `{frames, spikes, active}` (after `record off` the frames are kept for download, `active` is false, until the next `record on`) |
| `genome` | `{level, seed, growing: {level, seed, secs[, reason: "parts", parts]} or null, survival: {running, results: [{name, ok, fragile, seeds, readouts: [{label, hz, lo, hi, ok, per_seed[], seeds_out}], missing[]}], ok, tested, secs, where} or null, wiring: {edges_grown, synapses_grown, shared_connections_fraction} or null, rules: {level, groups, pairs, numbers, rank} or null, error, parts: {on, status}}`: the current fly's genome (see `grow` and `parts`). Each experiment runs on five seeds: `hz` is the mean, `ok` compares the mean with the range (allowing max(1 Hz, 15 %) above the top, SCIENCE.md section 2), `seeds_out` counts the seeds that miss on their own (by the same rule), and `fragile` marks an experiment that passes on the mean while some seed misses. `where` says how the re-test ran: `process` (a separate low-priority process, the default), `thread` (in the game's process: when no child process can be started or the child fails, when the game was made with `retest="thread"` or with its own brain factory and no `brain_kwargs`, or when the connectome has no file on disk) or `cache` (this fly was tested with the same settings before), and `secs` how long it took. `missing` (present only when non-empty) names the populations this fly does not have: their readouts are left out, and `ok` is `null` (n/a) when the experiment cannot be done without them. `parts.status` is `null` when the parts list is off, else `{tone: {dopamine: {mean, max, targets_on}, octopamine: {...}, serotonin: {...}}, graded_active, graded, modulatory, targets, local: [{spec, mode, release: {<compartment>: 0-1}}]}` (`mean` = the tone over that modulator's targets as a fraction of its full effect; `release` = how much of the whole cell's release a local neuron such as APL gives each compartment right now, weighted by its output synapses there, 1 when nothing is going on; `mode` is `regions` when the compartments are neuPrint's mushroom-body regions (calyx, pedunculus, each lobe; from `data/mb_roi_connectivity.json.gz`), `groups` when they are the Kenyon-cell lobe systems) |

`world` fields: `food[] = {id, kind (sugar|bitter|water), x, y, r, amount}`; `obstacles[] = {id, x, y, r}`;
`odours[] = {id, odour, x, y, strength, food}` (sources); `puffs[] = [x, y, r, c, odour]` (plume
filaments, up to 300); `wind = {angle, speed}` (direction the wind blows *toward*, rad; mm/s);
`female = null | {x, y, h, legs, receptive}`; `hand = null | [x, y]`; `tool` = current tool name;
`stripes = {count, phase, drum_speed}` (wall stripes: draw `count` alternating dark/bright bands around the wall, rotated by `phase` rad).

## Actions

### `POST /api/action` with a JSON body `{"type": ..., ...}` → `{"ok": true}` or `{"ok": false, "error": "..."}`

`{"ok": false, "error": "..."}` means nothing was done: an unknown or missing `type`, a missing field the
action needs, a field that is not a number where one is needed (text such as `"2.5"` is read as a number,
`null` as the default), `secs` of 0 or less, a negative `hz`, `factor` or `r`, or something the dish cannot
take (below). `{"ok": true}` means the action was queued for the next tick.

| type | fields | effect |
|---|---|---|
| `hand` | `x, y` | pointer position in mm (lure or hand tool) |
| `hand_off` | | pointer left the arena |
| `tool` | `tool` | `lure`, `hand`, `sugar`, `bitter`, `water`, `dust`, `shock`, `post`, an odour id, or `none` |
| `drop` | `kind, x, y[, food]` | drop `sugar`/`bitter`/`water`, a `post` (`r` optional, 4 mm), or an odour id (optionally with `food`: `sugar`, `bitter` or `water`). Refused outside the dish and within 2 mm of its wall (a post: within `r` + 1 mm); the page then shows why |
| `remove` | `id` | remove a food/obstacle/odour by id (refused if nothing has that id) |
| `dust` | `x, y` | puff dust (must be within 20 mm of the fly) |
| `shock` | `[secs]` | electric shock: drives PPL1 punishment dopamine |
| `sound` | `[secs]` | a loud sound (clap): Johnston's organ B neurons, which reach the giant fibre |
| `stripes` | `count, drum_speed` | paint `count` vertical stripes on the wall (0 = plain) and spin them at `drum_speed` rad/s (the optomotor drum) |
| `zap` | `spec, hz[, secs]` | stimulate a population (reply includes `n` neurons); `hz` 60 and `secs` 2 when left out |
| `silence` / `unsilence` | `spec` (unsilence: omit for all) | block / restore a population's output |
| `modulate` | `spec, factor` | scale a population's output (1 = normal) |
| `watch` / `unwatch` | `spec[, key]` / `key` | add / remove a custom readout (appears in `hz`). A `key` that names a built-in readout (`MN9`, `GF`, `DNp15L`, ...) is refused, since the decoder reads those; without a `key` the spec is the name, prefixed `watch:` if it collides. `unwatch` only removes custom watches. |
| `grow` | `level[, seed]` | grow a fly from the wiring rules (`type`, `class`, `bottleneck:K`) or go back to `real`; runs in the background (`state.genome.growing`), swaps the brain in when done and then tests every validated experiment on a private copy (`state.genome.survival` fills in) |
| `parts` | `on` (bool) | rebuild the current fly's brain with the parts list on or off (dopamine, octopamine and serotonin as slow tones; graded optic-lobe cells); the same background rebuild, swap and survival run as `grow`, on the same wiring. Refused while a rebuild or a growth is running, or if already in that state |
| `clear` | `[what]` | `all`, `food`, `odours`, `obstacles` |
| `reset` | | new fly, fresh brain (learned synapses and the checklist are kept) |
| `calm` | | reset the brain's activity to rest |
| `autopilot` | `on` | hand-built walking urge |
| `pause` | `on` | |
| `speed` | `value` | time scale 0.1..3 |
| `wind` | `angle, speed` | rad (direction it blows toward), mm/s (0 = off) |
| `female` | `on[, x, y]` | add / remove the female |
| `learning` | `on` and/or `forget: true` | toggle plasticity / reset learned weights |
| `scenario` | `id` or none | start a scenario / stop the running one |
| `record` | `on[, spikes]` | start/stop recording (frames; optionally every spike) |
| `state` | `hunger`, `thirst` | set internal state 0..1 |
| `place_fly` | `x, y[, h]` | teleport the fly |

## Queries

| endpoint | returns |
|---|---|
| `GET /api/types?q=LC10[&limit=50]` | `{types: [{type, n}]}` (a name the fly answers to as an alias, such as the female fly's `MN9` for FlyWire's `CB0701`, is listed too) |
| `GET /api/neuron?index=123` or `?body=<bodyId>` | `{neuron: {index, body_id, body_ref, type, side, superclass, class, subclass, nt, sign, nerve, neuromere, hex, soma, n_inputs, n_outputs, rate_hz, inputs[], outputs[], genes[], vfb, receptors, parts}}` (`body_ref` is the same id as a string: the female fly's FlyWire root ids are larger than JavaScript numbers hold exactly; `inputs`/`outputs`: `{type, side, synapses, connections, neurons, nt, sign}`; `vfb`: `null` or `{fbbt[], label, symbol, url, route, coarse, definition, breadcrumb: [{fbbt, label}], curated_nt[], evidence, peptides[], lineage[], birth, shared_by}`; `receptors`: `null` or `{class, label, depth, family, family_label, cluster, receptors: [{gene, extent, modulator, sign, flybase}]}`; `parts`: `null` when the parts list is off, else `{sign, modulator, keep_fast, graded, theta_mv[, curated: {action, curated[], predicted}][, local: {label, groups[], mode, compartments[]}][, receptor_fact: {receptors[], effects: {<modulator>: sign}, what, why}]}`) |
| `GET /api/ontology?q=lobula[&limit=30]` | `{q, classes: [{fbbt, label, symbol, types, neurons, spec}]}`: anatomy-ontology classes matching the text, largest first (`spec` is the `fbbt:` population) |
| `GET /api/ontology?id=FBbt_00003870` | `{class: {fbbt, label, symbol, definition, url, parents[], children: [{fbbt, label, types, neurons}], types: [{type, n}], n_types, neurons, spec, tags}}` (404 if the class is not among the kit's) |
| `GET /api/partners?spec=MN9&dir=in\|out[&top=15]` | `{spec, n, dir, rows: [{type, side, synapses, connections, neurons, nt, sign, fraction}]}` (`fraction` is a share of the partner's own traffic: with `dir=in` the share of the partner's output that goes to `spec`, with `dir=out` the share of its input that comes from `spec`) |
| `GET /api/trace?from=LC10a/L&to=DNa02/L[&hops=4&top=8&avoid=SPEC]` | `{paths: [{nodes[], score, net_sign, hops: [{src, dst, synapses, fraction, sign, src_size, dst_size}]}], neurons: [[neuron index per node or null]], relays: [[name, score]], secs}` (takes 0.5-3 s) |
| `GET /api/history?keys=MN9,GF[&n=400]` | `{bin_ms, history: {key: [hz...]}}` one value per tick |
| `GET /api/learning` | `{learning: {MBONtype: {strength, valence, nt, dopamine, kc_synapses}}, settings, depressed_fraction}` (`kc_synapses` and `settings.plastic_synapses` count KC→MBON connections, neuron pairs with one learned weight each, not synapses: 33,496 connections carrying 402,850 synapses in MaleCNS; `depressed_fraction` is a share of those connections) |
| `GET /api/decoder` | the decoder's DN→motor-pool table |
| `GET /api/recording` | JSON download of the recorded frames |
| `GET /api/spikes` | npz download of recorded spikes (`time_ms`, `neuron`, `body_id`) |

### `GET /api/genome`

The genome levels plus the current `state.genome` block.

### `GET /api/parts`

`{on, tables, counts, status}`: the layout's `parts` block plus whether the parts list is on and,
if so, the tones right now (`state.genome.parts.status`).

### `GET /api/genes`

The layout's `genetics` block on its own.

### `GET /api/lines?spec=pIP10&n=4`

Driver lines whose expression images match the population's neurons, from Janelia's NeuronBridge
(needs internet; results are cached under `data/neuronbridge/`). At most `n` (1-8) neurons are
searched, spread over the population. `{spec, n, sampled: [bodyId], unmatched: [bodyId],
lines: [{line, library, score, neurons, split}], version}`; `score` is NeuronBridge's colour-depth
match score, `neurons` how many of the searched neurons the line matched, `split` whether it is a
split-GAL4 line. 400 for a bad spec, and on the female fly (NeuronBridge matches MaleCNS neurons only); 502 when NeuronBridge cannot be reached.

### `GET /api/driver?line=SS02385`

MaleCNS neurons a driver line labels (its first `searched` images, brain images first, since a
line can have dozens): `{line, library, images, searched, neurons: [{body, index,
type, side, nb_type, score, in_kit}], spec, version}`. `nb_type` is the cell type NeuronBridge
holds (an earlier MaleCNS version), `type` the kit's for the same body, `in_kit` whether the body
exists in this data; `spec` (`body:...,body:...`) selects the best matches for `zap`, `silence` or
`watch`. 400 for an unknown line, and on the female fly; 502 when NeuronBridge cannot be reached.

`GET /api/neuron` additionally returns `genes: [{symbol, flybase, why}]`: *fru* and *dsx* when the
neuron is annotated as expressing them, and the synthesis/transport genes of its transmitter.
