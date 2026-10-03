// arena3d.js — the petri dish in three dimensions: the same world as arena.js (floor, wall, posts, food, odour plumes,
// the lure, the hand) with both flies as NeuroMechFly meshes. Nothing here runs until the page's "3-D" button is pressed:
// three.js (vendored under vendor/three/, loaded through the import map) and the model files under models/ are fetched
// only then, so the default page costs nothing new.
//
// What is real and what is not (docs/TWO_FLIES_PLAN.md 7.3, decision 19; the badge on the canvas says the same):
//   * the legs are an ANIMATION, not physics: for each frame the gait phase of the drawn body (the fly's `legs`) picks a
//     pose out of a gait atlas built by tools/build_fly_model.py, which replays NeuroMechFly's recorded stride through the
//     kit's own leg controller (standing still: the standing pose);
//   * both flies, the male included, wear NeuroMechFly's body, a model built from a female fly; a tint tells the sexes apart;
//   * wing extension, the abdomen's bend and the proboscis are HAND-BUILT rotations of those parts about hinges the atlas
//     names (the 1.2.1 model has no joints for them); a jump lifts the body by a hand-built amount;
//   * in live mode the flies are drawn at the drawn scale, about three times real size, like the 2-D dish and the senses.
"use strict";

const TAU = 2 * Math.PI;
// the drawn fly is FLY_HALF = 3.6 mm long from the centre (body.py), about three times real size: the meshes are scaled so
// that the standing fly is that long; the 2-D dish draws her a little larger than him (arena.js: 1.32 against 1.25)
const DRAWN_FLY_LENGTH_MM = 7.2;
const SEX_SIZE = { male: 1.0, female: 1.32 / 1.25 };
// colours by part, chosen by hand after a look at the real animal (nothing in the data says what colour a mesh is): red
// compound eyes, a tan head and thorax, an abdomen banded tan and dark with a dark tip on the male, clear wings, darker legs
// and tarsi, dark aristae; the female a shade lighter and greyer overall, as the 2-D dish tells the sexes apart by tint
const PALETTE = {
  male:   { head: 0xc4a06a, eye: 0xb8261b, thorax: 0xa98052, haltere: 0xd9cba3, band_light: 0xc9a266, band_dark: 0x6f5134,
            tip: 0x3b2a1f, wing: 0xdfe8f0, leg: 0x9a7a4f, tarsus: 0x5a4330, antenna: 0xb08b5e, arista: 0x3a2d22, proboscis: 0xb5905f },
  female: { head: 0xd2b283, eye: 0xc0352a, thorax: 0xbb976c, haltere: 0xe3d7b5, band_light: 0xd8b67d, band_dark: 0x8a6a48,
            tip: 0x7a5e42, wing: 0xe6edf3, leg: 0xad8f63, tarsus: 0x6e5640, antenna: 0xc19d72, arista: 0x4a3b2e, proboscis: 0xc6a373 },
};
/** Which palette entry a mesh takes, by its MuJoCo name (the atlas's part groups give the same answer; the names are quicker). */
function partColour(name) {
  if (/Eye$/.test(name)) return "eye";
  if (name === "Head") return "head";
  if (/Haltere$/.test(name)) return "haltere";
  if (name === "Thorax") return "thorax";
  if (/Wing$/.test(name)) return "wing";
  if (name === "A6") return "tip";
  if (/^A[1-5]/.test(name)) return (name === "A3" || name === "A5") ? "band_dark" : "band_light";   // A1A2, A4 light; A3, A5 dark
  if (/Arista$/.test(name)) return "arista";
  if (/Pedicel$|Funiculus$/.test(name)) return "antenna";
  if (/Rostrum$|Haustellum$/.test(name)) return "proboscis";
  if (/Tarsus[2-5]$/.test(name)) return "tarsus";
  if (/Coxa$|Femur$|Tibia$|Tarsus1$/.test(name)) return "leg";
  return "thorax";
}
const WING_OPEN_RAD = 75 * Math.PI / 180;        // a fully extended wing swings this far out (hand-built)
const WING_SING_RAD = 0.12;                       // a singing wing vibrates this much at 60 Hz, as the 2-D dish shows it
const ABDOMEN_BEND_RAD = 50 * Math.PI / 180;      // a courtship bend curls the abdomen down by this much (hand-built)
const PROBOSCIS_RAD = 60 * Math.PI / 180;         // the proboscis swings down and forward by this much when fully out (hand-built)
const JUMP_LIFT_MM = 3.0;                         // a jump lifts the body this much at its peak (hand-built, drawn scale)
const WALL_HEIGHT_MM = 3.0;
const MAX_PUFFS = 512, MAX_PARTICLES = 400;
const FOOD_COLS = { sugar: 0xf2b134, bitter: 0xa77bff, water: 0x4da3ff };

export const BADGE = "3-D animation: the legs follow the gait phase; not physics. Both flies use NeuroMechFly's body, " +
  "built from a female fly; the legs replay NeuroMechFly's recorded stride";

/** Why the 3-D view cannot run here, or null: three.js needs WebGL2 (since r163). Checked on a throw-away canvas before
 *  anything is fetched, so a browser without it stays in 2-D at no cost. */
export function webgl2Reason() {
  try {
    const c = document.createElement("canvas");
    const gl = c.getContext("webgl2");
    if (!gl) return "This browser gives no WebGL2 context, which the 3-D view needs: the dish stays in 2-D.";
    const ext = gl.getExtension("WEBGL_lose_context");
    if (ext) ext.loseContext();
    return null;
  } catch (e) {
    return `WebGL2 is unavailable here (${e && e.message ? e.message : e}): the dish stays in 2-D.`;
  }
}

function fetchJSON(url) { return fetch(url, { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); }); }
function fetchBin(url) { return fetch(url, { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.arrayBuffer(); }); }

/** The 3-D dish. Built by :func:`Arena3D.create` (which loads three.js and the model); the same interface as arena.js's
 *  Arena, so app.js's pointer handlers, tools and zoom keep working: resize(), draw(dt, view), setZoom(z), the `zoom`
 *  property, C2W(px, py) (a ray onto the floor), puff(x, y), shock(), clap(). */
export class Arena3D {
  /** Load three.js and the model (only now), check WebGL2 first, and build the dish. Throws with a plain message. */
  static async create(canvas, stage, L, ui = {}) {
    const reason = webgl2Reason();
    if (reason) throw new Error(reason);
    const THREE = await import("three");
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
    const loader = new GLTFLoader();
    const [gltf, atlas, bin] = await Promise.all([
      loader.loadAsync("models/nmf_fly.glb"), fetchJSON("models/nmf_gait.json"), fetchBin("models/nmf_gait.bin")]);
    return new Arena3D(THREE, OrbitControls, canvas, stage, L, ui, gltf, atlas, bin);
  }

  constructor(THREE, OrbitControls, canvas, stage, L, ui, gltf, atlas, bin) {
    this.THREE = THREE; this.canvas = canvas; this.stage = stage; this.L = L; this.ui = ui;
    this.R = L.arena_r || 50; this.w = this.h = 600; this.dpr = 1;
    this.zoom = 1; this.cameraMode = "top";
    this.lost = false; this.lostAt = 0; this.shockUntil = 0; this.clapAt = 0; this.on = false;
    this.frames = 0; this.fpsAt = performance.now(); this.fps = 0;
    this.odourColour = {}; for (const o of L.odours || []) this.odourColour[o.id] = o.colour;
    // ---- the renderer, the scene, the camera
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x141b25);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.5, 2000);
    this.camera.up.set(0, 0, 1);                                        // z up: MuJoCo's frame, 1 unit = 1 mm
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.12; this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 6; this.controls.maxDistance = this.R * 4;
    this.controls.enablePan = false;
    this.controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };   // the left button keeps the tools
    this.controls.touches = { ONE: null, TWO: THREE.TOUCH.DOLLY_ROTATE };
    this.controls.addEventListener("change", () => { this.zoom = this._zoomFromDistance(); });
    this.baseDistance = this.R * 2.4;
    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); this.lost = true; this.lostAt = performance.now(); this._say("graphics reset, restoring…"); });
    canvas.addEventListener("webglcontextrestored", () => { this.lost = false; this._say(""); });
    this._buildWorld();
    this._buildModel(gltf, atlas, bin);
    this.flies = new Map();                                               // fly id (or "female") -> its meshes
    this.setCamera("top");
    this.resize();
  }

  // ---------------------------------------------------------------- the Arena interface
  resize() {
    const st = this.stage.getBoundingClientRect();
    this.w = Math.max(280, Math.floor(st.width || 280)); this.h = Math.max(280, Math.floor(st.height || st.width || 280));
    this.renderer.setSize(this.w, this.h, false);
    this.canvas.style.width = this.w + "px"; this.canvas.style.height = this.h + "px";
    this.camera.aspect = this.w / this.h; this.camera.updateProjectionMatrix();
  }
  /** Zoom (1 = the whole dish, up to 4x) as the camera's distance; above 1 the camera follows the focused fly. */
  setZoom(z) {
    this.zoom = Math.max(1, Math.min(4, z));
    const d = this.baseDistance / this.zoom, cur = this.camera.position.clone().sub(this.controls.target);
    if (cur.length() > 1e-6) { cur.setLength(d); this.camera.position.copy(this.controls.target).add(cur); }
    this.controls.update();
    return this.zoom;
  }
  _zoomFromDistance() {
    const d = this.camera.position.distanceTo(this.controls.target);
    return Math.max(1, Math.min(4, this.baseDistance / Math.max(1e-6, d)));
  }
  /** Canvas pixels to world mm: a ray from the camera through the pixel onto the floor plane (z = 0). */
  C2W(px, py) {
    const THREE = this.THREE, ndc = new THREE.Vector2((px / this.w) * 2 - 1, -(py / this.h) * 2 + 1);
    const ray = new THREE.Raycaster(); ray.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    if (ray.ray.intersectPlane(this._floorPlane, hit)) return [hit.x, hit.y];
    return [1e6, 1e6];                                                  // the sky: nowhere in the dish
  }
  puff(x, y) {
    for (let i = 0; i < 70; i++) {
      const a = Math.random() * TAU, v = 4 + Math.random() * 14;
      this.particles.push({ x, y, z: 0.5 + Math.random(), vx: Math.cos(a) * v, vy: Math.sin(a) * v, vz: 2 + Math.random() * 6, life: 0.8 + Math.random() * 0.7, age: 0 });
    }
    while (this.particles.length > MAX_PARTICLES) this.particles.shift();
  }
  shock() { this.shockUntil = performance.now() + 500; }
  clap() { this.clapAt = performance.now(); }

  /** The camera presets: top (the whole dish from above), follow (behind and above the focused fly), side. */
  setCamera(mode) {
    this.cameraMode = mode;
    const R = this.R, c = this.camera, t = this.controls.target;
    if (mode === "side") { t.set(0, 0, 2); c.position.set(R * 1.7, -R * 0.9, R * 0.55); }
    else if (mode === "follow") { t.set(0, 0, 1.5); c.position.set(-18, 0, 11); }
    else { t.set(0, 0, 0); c.position.set(0, -R * 0.02, this.baseDistance); }  // a hair off the pole, so the orbit controls keep their bearings
    this.controls.update();
    this.zoom = this._zoomFromDistance();
  }

  // ---------------------------------------------------------------- the world
  _buildWorld() {
    const THREE = this.THREE, R = this.R, scene = this.scene;
    this._floorPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    scene.add(new THREE.HemisphereLight(0xdfe8f5, 0x1a2230, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(40, -30, 70); scene.add(sun);
    const floor = new THREE.Mesh(new THREE.CircleGeometry(R, 128), new THREE.MeshStandardMaterial({ color: 0x222d3b, roughness: 0.95, metalness: 0.0 }));
    scene.add(floor);
    for (let k = 1; k < 4; k++) {                                        // the faint rings of the 2-D dish
      const g = new THREE.BufferGeometry().setFromPoints(Array.from({ length: 97 }, (_, i) => new THREE.Vector3(Math.cos(i / 96 * TAU) * R * k / 4, Math.sin(i / 96 * TAU) * R * k / 4, 0.02)));
      scene.add(new THREE.LineLoop(g, new THREE.LineBasicMaterial({ color: 0x2c3a4c })));
    }
    const wallGeo = new THREE.CylinderGeometry(R, R, WALL_HEIGHT_MM, 128, 1, true); wallGeo.rotateX(Math.PI / 2);
    const wall = new THREE.Mesh(wallGeo, new THREE.MeshStandardMaterial({ color: 0x3a4a60, roughness: 0.8, side: THREE.DoubleSide, transparent: true, opacity: 0.55 }));
    wall.position.z = WALL_HEIGHT_MM / 2; scene.add(wall);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(R, 0.25, 8, 128), new THREE.MeshStandardMaterial({ color: 0x4a5a72, roughness: 0.6 }));
    rim.position.z = WALL_HEIGHT_MM; scene.add(rim);
    // things that come and go: posts, food, odour sources, puffs, particles, the pointer, effects
    this.posts = new Map(); this.food = new Map(); this.odours = new Map();
    this.postGeo = new THREE.CylinderGeometry(1, 1, 4, 24); this.postGeo.rotateX(Math.PI / 2);
    this.postMat = new THREE.MeshStandardMaterial({ color: 0x3f4d63, roughness: 0.7 });
    this.foodGeo = new THREE.CylinderGeometry(1, 1, 0.35, 32); this.foodGeo.rotateX(Math.PI / 2);
    this.ringGeo = new THREE.RingGeometry(0.85, 1, 48);
    this.puffs = this._points(MAX_PUFFS, 3.2, 0x9be15d, 0.28);
    this.dust = this._points(MAX_PARTICLES, 1.6, 0xd6dee8, 0.7);
    this.particles = [];
    this.pointerRing = new THREE.Mesh(new THREE.RingGeometry(2.4, 2.9, 48), new THREE.MeshBasicMaterial({ color: 0x9be15d, transparent: true, opacity: 0.7, side: THREE.DoubleSide }));
    this.pointerRing.position.z = 0.05; this.pointerRing.visible = false; scene.add(this.pointerRing);
    this.lure = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshStandardMaterial({ color: 0x11151b, roughness: 0.5 }));
    this.lure.scale.set(1.2, 0.55, 0.5); this.lure.visible = false; scene.add(this.lure);
    this.shockLight = new THREE.PointLight(0xffd166, 0, 40); this.shockLight.position.z = 6; scene.add(this.shockLight);
    this.clapRings = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.RingGeometry(0.92, 1, 64), new THREE.MeshBasicMaterial({ color: 0x8fb8ff, transparent: true, opacity: 0, side: THREE.DoubleSide })); m.position.z = 0.06; m.visible = false; scene.add(m); return m; });
    this.drum = null; this.drumCount = 0;
  }
  _points(n, size, colour, opacity) {
    const THREE = this.THREE, geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setDrawRange(0, 0);
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size, vertexColors: true, transparent: true, opacity, depthWrite: false, sizeAttenuation: true }));
    this.scene.add(pts);
    return pts;
  }
  _say(text) { const el = this.ui.msg; if (!el) return; el.textContent = text; el.hidden = !text; }

  // ---------------------------------------------------------------- the model and the gait atlas
  _buildModel(gltf, atlas, bin) {
    const THREE = this.THREE;
    this.atlas = atlas;
    this.G = atlas.geoms.length;
    this.frameCount = atlas.frames || (atlas.phases + 1);
    this.phases = atlas.phases; this.standing = atlas.standing == null ? atlas.phases : atlas.standing;
    const want = this.frameCount * this.G * 7;
    const data = new Float32Array(bin);
    if (data.length !== want) throw new Error(`models/nmf_gait.bin holds ${data.length} numbers, not ${want} (${this.frameCount} frames x ${this.G} geoms x 7)`);
    this.A = data;
    const unit = atlas.unit === "m" ? 1000 : atlas.unit === "cm" ? 10 : 1;   // the page wants mm
    this.unitScale = unit;
    this.template = gltf.scene;
    // how long the standing model is along its own x: the drawn fly is DRAWN_FLY_LENGTH_MM, so the meshes are scaled to that
    const box = new THREE.Box3().setFromObject(this.template);
    const len = Math.max(1e-6, (box.max.x - box.min.x) * unit);
    this.modelScale = DRAWN_FLY_LENGTH_MM / len;
    this.floorLift = -box.min.z * unit;                                  // the feet of the standing fly touch z = 0
    this.modelLengthMm = len;
    // the parts the hand-built rotations move, as sets of geom indices, and their hinges
    const index = new Map(atlas.geoms.map((n, i) => [n, i]));
    const part = (names) => (names || []).map((n) => index.get(n)).filter((i) => i != null);
    const P = atlas.parts || {};
    this.parts = { wingL: part(P.wing_left), wingR: part(P.wing_right), abdomen: part(P.abdomen), proboscis: part(P.proboscis),
                   foreLegs: [...part(P.legs && P.legs.LF), ...part(P.legs && P.legs.RF)] };
    const H = atlas.hinges || {};
    const hinge = (k) => H[k] ? new THREE.Vector3(H[k][0], H[k][1], H[k][2]) : new THREE.Vector3();   // the model's unit, like the atlas
    this.hinges = { wingL: hinge("wing_left"), wingR: hinge("wing_right"), abdomen: hinge("abdomen"), proboscis: hinge("proboscis") };
    this.missing = new Set();
    this.triangles = atlas.triangles || 0;
    // scratch
    this._p0 = new THREE.Vector3(); this._p1 = new THREE.Vector3(); this._q0 = new THREE.Quaternion(); this._q1 = new THREE.Quaternion();
    this._rot = new THREE.Quaternion(); this._axis = new THREE.Vector3();
  }
  /** One fly's meshes: a clone of the model with its own tinted material, its geom nodes by atlas order. */
  _makeFly(key, sex) {
    const THREE = this.THREE, group = new THREE.Group();
    const body = this.template.clone(true);
    const pal = PALETTE[sex] || PALETTE.male, mats = {};
    const material = (kind) => mats[kind] || (mats[kind] = kind === "wing"
      ? new THREE.MeshStandardMaterial({ color: pal.wing, roughness: 0.35, metalness: 0.1, transparent: true, opacity: 0.38, side: THREE.DoubleSide, depthWrite: false })
      : kind === "eye" ? new THREE.MeshStandardMaterial({ color: pal.eye, roughness: 0.4, metalness: 0.0 })
      : new THREE.MeshStandardMaterial({ color: pal[kind], roughness: 0.62, metalness: 0.05 }));
    body.traverse((o) => { if (o.isMesh) { o.material = material(partColour(o.name)); o.frustumCulled = false; } });
    body.scale.setScalar(this.unitScale);
    group.add(body);
    const nodes = this.atlas.geoms.map((name) => { const n = body.getObjectByName(name); if (!n && !this.missing.has(name)) { this.missing.add(name); console.warn(`3-D dish: the model has no node named ${name}`); } return n || null; });
    group.scale.setScalar(this.modelScale * (SEX_SIZE[sex] || 1));
    this.scene.add(group);
    const fly = { key, sex, group, body, nodes, mats };
    this.flies.set(key, fly);
    return fly;
  }
  _frameOf(f) {
    // the atlas frame from the drawn body's gait phase, as arena.js turns `legs` into the swing of each leg
    if (!f.walking) return [this.standing, this.standing, 0];
    const phase = (((f.legs || 0) * 1.6) % TAU + TAU) % TAU / TAU, fi = phase * this.phases;
    const i0 = Math.floor(fi) % this.phases, i1 = (i0 + 1) % this.phases;
    return [i0, i1, fi - Math.floor(fi)];
  }
  /** Place a fly's geom nodes for this frame: the atlas pose at its gait phase (interpolated), then the hand-built
   *  rotations of the wings, the abdomen and the proboscis about their hinges, then the body's pose in the dish. */
  _poseFly(fly, f, o) {
    const THREE = this.THREE, A = this.A, G = this.G, [i0, i1, t] = this._frameOf(f);
    const p0 = this._p0, p1 = this._p1, q0 = this._q0, q1 = this._q1, now = performance.now() / 1000;
    const wl = f.wingL || 0, wr = f.wingR || 0, ab = f.abdomen || 0, pr = f.prob || 0;
    const mode = f.mode || "walk", fem = fly.sex === "female";
    const song = mode !== "escape" && f.jump == null && (wl > 0.35 || wr > 0.35) && (!fem || o.sim);
    const rots = [];                                                     // [indices set, hinge, quaternion] per moving part
    // hand-built: a wing swings outward from pointing back (-x) about the z axis at its root (the left wing, on MuJoCo's +y
    // side, by a negative turn; the right one positive), a singing wing shivers; the abdomen curls down about the y axis at
    // its base; the proboscis swings down and forward about the y axis at its base
    if (this.parts.wingL.length && wl > 0.005) rots.push([new Set(this.parts.wingL), this.hinges.wingL, this._rot.clone().setFromAxisAngle(this._axis.set(0, 0, 1), -(WING_OPEN_RAD * wl + (song && wl > 0.35 ? WING_SING_RAD * Math.sin(now * 60) : 0)))]);
    if (this.parts.wingR.length && wr > 0.005) rots.push([new Set(this.parts.wingR), this.hinges.wingR, this._rot.clone().setFromAxisAngle(this._axis.set(0, 0, 1), WING_OPEN_RAD * wr + (song && wr > 0.35 ? WING_SING_RAD * Math.sin(now * 60) : 0))]);
    if (this.parts.abdomen.length && ab > 0.005) rots.push([new Set(this.parts.abdomen), this.hinges.abdomen, this._rot.clone().setFromAxisAngle(this._axis.set(0, 1, 0), -ABDOMEN_BEND_RAD * ab)]);
    if (this.parts.proboscis.length && pr > 0.02) rots.push([new Set(this.parts.proboscis), this.hinges.proboscis, this._rot.clone().setFromAxisAngle(this._axis.set(0, 1, 0), PROBOSCIS_RAD * pr)]);
    for (let g = 0; g < G; g++) {
      const node = fly.nodes[g]; if (!node) continue;
      const a0 = (i0 * G + g) * 7, a1 = (i1 * G + g) * 7;
      p0.set(A[a0], A[a0 + 1], A[a0 + 2]); p1.set(A[a1], A[a1 + 1], A[a1 + 2]);
      q0.set(A[a0 + 4], A[a0 + 5], A[a0 + 6], A[a0 + 3]); q1.set(A[a1 + 4], A[a1 + 5], A[a1 + 6], A[a1 + 3]);   // MuJoCo w,x,y,z -> three x,y,z,w
      p0.lerp(p1, t); q0.slerp(q1, t);                                   // in the model's unit: the body node scales it to mm
      for (const [set, hinge, rq] of rots) {
        if (!set.has(g)) continue;
        p0.sub(hinge).applyQuaternion(rq).add(hinge);                  // p' = hinge + R (p - hinge)
        q0.premultiply(rq);                                              // q' = R q
      }
      node.position.copy(p0); node.quaternion.copy(q0);
    }
    const jump = f.jump == null ? 0 : Math.sin(Math.PI * f.jump);
    fly.group.position.set(f.x, f.y, this.floorLift * fly.group.scale.x + JUMP_LIFT_MM * jump);
    fly.group.rotation.set(0, 0, f.h || 0);
    fly.group.visible = true;
  }

  // ---------------------------------------------------------------- per frame
  draw(dt, view) {
    if (document.visibilityState === "hidden") return;                  // nothing drawn while the tab is hidden
    const { S, pose, female, pointer, tool, stripes } = view;
    const flies = view.flies || (pose ? [pose] : []);
    const THREE = this.THREE, w = (S && S.world) || {};
    // the flies: every simulated one, and the scripted female of single-fly play
    for (const fly of this.flies.values()) fly.group.visible = false;
    for (const f of flies) {
      const key = f.id == null ? 0 : f.id, sex = f.sex || this.L.sex || "male";
      const fly = this.flies.get(key) || this._makeFly(key, sex);
      this._poseFly(fly, f, { sim: true });
    }
    if (female) this._poseFly(this.flies.get("female") || this._makeFly("female", "female"), { ...female, walking: !!female.walking }, { sim: false });
    // the camera: follows the focused fly above zoom 1 or in the follow preset
    this._camera(dt, pose);
    this._world(dt, w, stripes);
    this._pointer(pointer, tool, w, view.handAng || 0);
    this._effects(pose);
    this.controls.update();
    if (!this.lost) this.renderer.render(this.scene, this.camera);
    this._stats();
  }
  _camera(dt, pose) {
    const t = this.controls.target, c = this.camera;
    if (this.cameraMode === "follow" && pose) {
      const nx = pose.x, ny = pose.y, dx = nx - t.x, dy = ny - t.y;        // the target rides with the fly; the camera keeps its offset
      t.x = nx; t.y = ny; c.position.x += dx; c.position.y += dy;
    } else if (this.cameraMode === "top" && this.zoom > 1 && pose) {
      const k = Math.min(1, dt * 4), lim = Math.max(0, this.R * (1 - 1 / this.zoom));
      let tx = pose.x, ty = pose.y; const d = Math.hypot(tx, ty); if (d > lim) { tx *= lim / d; ty *= lim / d; }
      const dx = (tx - t.x) * k, dy = (ty - t.y) * k; t.x += dx; t.y += dy; c.position.x += dx; c.position.y += dy;
    } else if (this.cameraMode === "top" && this.zoom <= 1 && (t.x || t.y)) {
      const k = Math.min(1, dt * 4), dx = -t.x * k, dy = -t.y * k; t.x += dx; t.y += dy; c.position.x += dx; c.position.y += dy;
    }
  }
  _world(dt, w, stripes) {
    const THREE = this.THREE;
    this._sync(this.posts, w.obstacles || [], (o) => { const m = new THREE.Mesh(this.postGeo, this.postMat); this.scene.add(m); return m; },
      (m, o) => { m.position.set(o.x, o.y, 2); m.scale.set(o.r, o.r, 1); });
    this._sync(this.food, w.food || [], (f) => { const m = new THREE.Mesh(this.foodGeo, new THREE.MeshStandardMaterial({ color: FOOD_COLS[f.kind] || FOOD_COLS.sugar, roughness: 0.4 })); this.scene.add(m); return m; },
      (m, f) => { m.position.set(f.x, f.y, 0.18); m.scale.set(f.r, f.r, 1); m.material.opacity = 1; });
    this._sync(this.odours, w.odours || [], (o) => { const m = new THREE.Mesh(this.ringGeo, new THREE.MeshBasicMaterial({ color: new THREE.Color(this.odourColour[o.odour] || "#9be15d"), transparent: true, opacity: 0.8, side: THREE.DoubleSide })); m.position.z = 0.04; this.scene.add(m); return m; },
      (m, o) => { m.position.set(o.x, o.y, 0.04); m.scale.set(2.4, 2.4, 1); });
    // puffs: points coloured by odour, faded by strength
    const puffs = w.puffs || [], pp = this.puffs.geometry.attributes.position, pc = this.puffs.geometry.attributes.color, col = new THREE.Color();
    let n = 0;
    for (const p of puffs) { if (n >= MAX_PUFFS) break; col.set(this.odourColour[p[4]] || "#9be15d"); const s = Math.min(1, p[3]) * 0.9 + 0.1; pp.setXYZ(n, p[0], p[1], 0.6 + p[2] * 0.4); pc.setXYZ(n, col.r * s, col.g * s, col.b * s); n++; }
    pp.needsUpdate = pc.needsUpdate = true; this.puffs.geometry.setDrawRange(0, n);
    // dust particles of the puff tool
    const dp = this.dust.geometry.attributes.position, dc = this.dust.geometry.attributes.color, ps = this.particles;
    let m = 0;
    for (let i = ps.length - 1; i >= 0; i--) {
      const q = ps[i]; q.age += dt; q.x += q.vx * dt; q.y += q.vy * dt; q.z += q.vz * dt; q.vx *= 0.93; q.vy *= 0.93; q.vz = q.vz * 0.9 - 3 * dt;
      if (q.age > q.life) { ps.splice(i, 1); continue; }
      const a = 1 - q.age / q.life; dp.setXYZ(m, q.x, q.y, Math.max(0.1, q.z)); dc.setXYZ(m, 0.84 * a, 0.87 * a, 0.91 * a); m++;
    }
    dp.needsUpdate = dc.needsUpdate = true; this.dust.geometry.setDrawRange(0, m);
    // the optomotor drum: bands painted on the inside of the wall, spun by the server's phase
    const count = stripes && stripes.count > 0 ? stripes.count : 0;
    if (count !== this.drumCount) {
      if (this.drum) { this.scene.remove(this.drum); this.drum.traverse((o) => { if (o.geometry) o.geometry.dispose(); }); this.drum = null; }
      this.drumCount = count;
      if (count) {
        this.drum = new THREE.Group();
        const step = TAU / count, dark = new THREE.MeshBasicMaterial({ color: 0x0d1218, side: THREE.DoubleSide }), bright = new THREE.MeshBasicMaterial({ color: 0xcfd8e3, side: THREE.DoubleSide });
        for (let i = 0; i < count; i++) {
          const g = new THREE.CylinderGeometry(this.R - 0.3, this.R - 0.3, WALL_HEIGHT_MM + 0.4, 12, 1, true, i * step, step); g.rotateX(Math.PI / 2);
          const band = new THREE.Mesh(g, i % 2 ? bright : dark); band.position.z = WALL_HEIGHT_MM / 2 + 0.2; this.drum.add(band);
        }
        this.scene.add(this.drum);
      }
    }
    if (this.drum) this.drum.rotation.z = stripes.phase || 0;
  }
  /** Keep a map of meshes in step with a list of world objects (by id or position): make, update, remove. */
  _sync(map, items, make, update) {
    const seen = new Set();
    items.forEach((o, i) => { const key = o.id != null ? o.id : `${i}:${o.x},${o.y}`; seen.add(key); let m = map.get(key); if (!m) { m = make(o); map.set(key, m); } update(m, o); });
    for (const [key, m] of map) if (!seen.has(key)) { this.scene.remove(m); if (m.material && m.material !== this.postMat) m.material.dispose(); map.delete(key); }
  }
  _pointer(pointer, tool, w, handAng) {
    let x = null, y = null, kind = tool;
    if (pointer && pointer.inside) { x = pointer.x; y = pointer.y; }
    else if (w.hand && (w.tool === "lure" || w.tool === "hand")) { x = w.hand[0]; y = w.hand[1]; kind = w.tool; }
    const show = x != null && Math.hypot(x, y) < this.R;
    this.pointerRing.visible = show && kind !== "lure"; this.lure.visible = show && kind === "lure";
    if (!show) return;
    if (kind === "lure") { this.lure.position.set(x, y, 0.5); this.lure.rotation.set(0, 0, -(pointer && pointer.ang != null ? pointer.ang : handAng)); }
    else { this.pointerRing.position.set(x, y, 0.05); this.pointerRing.material.color.set(kind === "hand" ? "#e7edf4" : kind === "shock" ? "#ffd166" : kind === "dust" ? "#c9d4e2" : this.odourColour[kind] || "#9be15d"); this.pointerRing.scale.setScalar(kind === "hand" ? 2 : 1); }
  }
  _effects(pose) {
    const now = performance.now();
    if (pose && now < this.shockUntil) { this.shockLight.position.set(pose.x, pose.y, 6); this.shockLight.intensity = 600 * (0.5 + 0.5 * Math.sin(now / 30)); } else this.shockLight.intensity = 0;
    const u = (now - this.clapAt) / 600;
    this.clapRings.forEach((m, j) => {
      if (!pose || u >= 1) { m.visible = false; return; }
      const r = 4 + 26 * ((u + j * 0.33) % 1); m.visible = true; m.position.set(pose.x, pose.y, 0.06); m.scale.setScalar(r); m.material.opacity = 0.7 * (1 - u);
    });
  }
  _stats() {
    this.frames++;
    const now = performance.now();
    if (now - this.fpsAt >= 1000) { this.fps = Math.round(this.frames * 1000 / (now - this.fpsAt)); this.frames = 0; this.fpsAt = now; }
    const info = this.renderer.info.render;
    window.__vf3d = { on: this.on, flies: [...this.flies.values()].filter((f) => f.group.visible).length, triangles: info.triangles, calls: info.calls,
                      fps: this.fps, lost: this.lost, missingNodes: this.missing.size, modelLengthMm: this.modelLengthMm, scale: this.modelScale, camera: this.cameraMode };
  }
  show() { this.on = true; this.canvas.hidden = false; this.resize(); this._stats(); }
  hide() { this.on = false; this.canvas.hidden = true; if (window.__vf3d) window.__vf3d.on = false; }
}
