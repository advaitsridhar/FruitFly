# Game server API

`python fly_game.py` starts a local HTTP server (default `http://127.0.0.1:8765`). The browser
page is served from `virtual_fly/web/`. Everything the page does goes through this API, so any
program that speaks HTTP can drive the fly.

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
| `readouts[]` | `{key, spec, label, group, max, colour}`: the key-neuron bars, in display order, grouped by `group` |
| `checks[]` | `{id, text}`: the experiments checklist |
| `odours[]` | `{id, name, glomeruli[], innate, colour, note}` |
| `scenarios[]` | `{id, name, description}` |
| `retina` | `{L: {n_az, n_el, az[], el[]}, R: {...}}` facet directions (radians, fly frame; `az` + = left) |
| `profile` | model profile name (`game`, `pure`, `brakes`) |
| `settings` | brain settings dict (dt, backend `numpy`/`numba`, gain, fatigue, silenced, plasticity ...) |
| `decoder` | per decoder DN spec: motor synapses it reaches (`direct_motor_synapses`, `two_hop_motor_synapses_by_neuromere`) |
| `columnar_vision` | bool: T4/T5 columns driven from the retina |
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
| `mode` | `idle`, `walk`, `feed`, `groom`, `escape`, `backward`, `court` |
| `fly` | `{x, y, h, v, w, mode, prob, legs, groom, wingL, wingR, abdomen, jump, hx, hy, dist}`: position mm, heading rad (0 = +x, CCW), forward speed mm/s, yaw rate rad/s, proboscis 0..1, gait phase, groom phase, wing extensions 0..1, abdomen bend 0..1, jump progress 0..1 or null, head position, distance walked |
| `world` | `{food[], obstacles[], odours[], puffs[], wind, female, hand, tool}`: see below |
| `senses` | which senses are active now: keys `taste_sugar`, `taste_bitter`, `taste_water`, `small` (`"L"`,`"R"`,`"LR"`), `loom`, `flow`, `smell` (odour id), `pheromone`, `courting`, `sound`, `wind` (bearing in degrees the wind comes from, + = left; `0` means straight ahead, so test for the key, not the value), `dust`, `touch`, `reward`, `shock`, `zap` (text) |
| `retina` | `{L: base64, R: base64}`: one byte per facet (0 dark .. 255 bright), facet order matches `layout.retina` |
| `hz` | firing rate (Hz per neuron, smoothed) per readout key, including custom watches |
| `motor` | decoder drives: `forward, yaw, backward, halt, feed, groom, song, court` (0..1; yaw -1..1, + = right) |
| `driver` | text explaining what drives the current behaviour |
| `spikes[]` | up to 2,500 neuron indices that spiked this tick (for the brain map) |
| `sps` | spikes per second in the whole brain |
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

`world` fields: `food[] = {id, kind (sugar|bitter|water), x, y, r, amount}`; `obstacles[] = {id, x, y, r}`;
`odours[] = {id, odour, x, y, strength, food}` (sources); `puffs[] = [x, y, r, c, odour]` (plume
filaments, up to 300); `wind = {angle, speed}` (direction the wind blows *toward*, rad; mm/s);
`female = null | {x, y, h, legs, receptive}`; `hand = null | [x, y]`; `tool` = current tool name;
`stripes = {count, phase, drum_speed}` (wall stripes: draw `count` alternating dark/bright bands around the wall, rotated by `phase` rad).

## Actions

### `POST /api/action` with a JSON body `{"type": ..., ...}` → `{"ok": true}` or `{"ok": false, "error": "..."}`

| type | fields | effect |
|---|---|---|
| `hand` | `x, y` | pointer position in mm (lure or hand tool) |
| `hand_off` | | pointer left the arena |
| `tool` | `tool` | `lure`, `hand`, `sugar`, `bitter`, `water`, `dust`, `shock`, `post`, an odour id, or `none` |
| `drop` | `kind, x, y[, food]` | drop `sugar`/`bitter`/`water`, a `post` (`r` optional), or an odour id (optionally with `food: "sugar"`) |
| `remove` | `id` | remove a food/obstacle/odour by id |
| `dust` | `x, y` | puff dust (must be within 20 mm of the fly) |
| `shock` | `[secs]` | electric shock: drives PPL1 punishment dopamine |
| `sound` | `[secs]` | a loud sound (clap): Johnston's organ A/B neurons, which reach the giant fibre |
| `stripes` | `count, drum_speed` | paint `count` vertical stripes on the wall (0 = plain) and spin them at `drum_speed` rad/s (the optomotor drum) |
| `zap` | `spec, hz[, secs]` | stimulate a population (reply includes `n` neurons) |
| `silence` / `unsilence` | `spec` (unsilence: omit for all) | block / restore a population's output |
| `modulate` | `spec, factor` | scale a population's output (1 = normal) |
| `watch` / `unwatch` | `spec[, key]` / `key` | add / remove a custom readout (appears in `hz`). A `key` that names a built-in readout (`MN9`, `GF`, `DNp15L`, ...) is refused, since the decoder reads those; without a `key` the spec is the name, prefixed `watch:` if it collides. `unwatch` only removes custom watches. |
| `clear` | `[what]` | `all`, `food`, `odours`, `obstacles` |
| `reset` | | new fly, fresh brain (learned synapses kept) |
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
| `GET /api/types?q=LC10[&limit=50]` | `{types: [{type, n}]}` |
| `GET /api/neuron?index=123` or `?body=<bodyId>` | `{neuron: {index, body_id, type, side, superclass, class, subclass, nt, sign, nerve, neuromere, hex, soma, n_inputs, n_outputs, rate_hz, inputs[], outputs[]}}` (`inputs`/`outputs`: `{type, side, synapses, connections, neurons, nt, sign}`) |
| `GET /api/partners?spec=MN9&dir=in\|out[&top=15]` | `{spec, n, dir, rows: [{type, side, synapses, connections, neurons, nt, sign, fraction}]}` |
| `GET /api/trace?from=LC10a/L&to=DNa02/L[&hops=4&top=8&avoid=SPEC]` | `{paths: [{nodes[], score, net_sign, hops: [{src, dst, synapses, fraction, sign, src_size, dst_size}]}], neurons: [[neuron index per node or null]], relays: [[name, score]], secs}` (takes 0.5-3 s) |
| `GET /api/history?keys=MN9,GF[&n=400]` | `{bin_ms, history: {key: [hz...]}}` one value per tick |
| `GET /api/learning` | `{learning: {MBONtype: {strength, valence, nt, dopamine, kc_synapses}}, settings, depressed_fraction}` |
| `GET /api/decoder` | the decoder's DN→motor-pool table |
| `GET /api/recording` | JSON download of the recorded frames |
| `GET /api/spikes` | npz download of recorded spikes (`time_ms`, `neuron`, `body_id`) |
