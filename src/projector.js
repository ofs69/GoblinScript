"use strict";
// The ONE definition of the VR viewport projection, shared by every tool that
// shows a VR aim: the training-ingest aiming page (vr_project.py serve) and
// goblinscript's prep page. Both render the FINAL video with ffmpeg's `v360`
// filter, so this is what a preview has to agree with, and there is exactly one
// copy of it so the two tools cannot drift apart.
//
// The mapping is v360's, pinned per-pixel against ffmpeg itself (equirect,
// hequirect and fisheye; every aim; roll included) to sub-texel agreement:
//
//   flat output pixel -> x = (2i+1)/w - 1, y = (2j+1)/h - 1   (+y is DOWN)
//   direction         -> normalize(x*tan(h_fov/2), y*tan(v_fov/2), 1)
//   rotated by         Ry(yaw) * Rp(pitch) * Rr(roll)
//   equirect          -> u = atan2(X,Z)/span + 0.5,  v = asin(Y)/PI + 0.5
//                        span = 2PI (equirect) or PI (hequirect, one 180 eye)
//   fisheye           -> rho = acos(Z) / (lens/2), uv = rho * normalize(X,Y)
//
// Aiming past the source's coverage (a 90 deg viewport at yaw -45 on a 180 deg
// lens) is not an error: v360 CLAMPS to the edge texel and smears it, and
// CLAMP_TO_EDGE sampling here reproduces that exactly -- the preview shows the
// same smear the render will produce. `coverage()` reports how much of the
// viewport is out there so the page can warn rather than let it pass unseen.

(function (root) {

const PROJ = { equirect: 0, hequirect: 1, fisheye: 2 };
const D2R = Math.PI / 180;

const VS = `
attribute vec2 aPos;
varying vec2 vNDC;
void main() {
  // v360's flat output counts +y DOWNWARD; clip space counts it up.
  vNDC = vec2(aPos.x, -aPos.y);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FS = `
precision highp float;
varying vec2 vNDC;
uniform sampler2D uTex;
uniform vec2  uTan;       // (tan(h_fov/2), tan(v_fov/2))
uniform mat3  uRot;       // Ry(yaw) * Rp(pitch) * Rr(roll)
uniform int   uInput;     // 0 equirect, 1 hequirect, 2 fisheye
uniform float uLensHalf;  // fisheye lens half-FOV, radians
uniform vec4  uRect;      // sub-rectangle of the texture holding the eye
const float PI = 3.141592653589793;

void main() {
  vec3 v = normalize(vec3(vNDC.x * uTan.x, vNDC.y * uTan.y, 1.0));
  vec3 d = uRot * v;
  vec2 uv;
  if (uInput == 2) {
    float rho = acos(clamp(d.z, -1.0, 1.0)) / uLensHalf;
    vec2 dir = d.xy / max(length(d.xy), 1e-9);
    uv = rho * dir * 0.5 + 0.5;
  } else {
    float phi = atan(d.x, d.z);
    float span = (uInput == 1) ? PI : 2.0 * PI;
    uv = vec2(phi / span + 0.5,
              asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5);
  }
  // clamp, not discard: v360 smears the edge texel and so must the preview
  uv = clamp(uv, 0.0, 1.0);
  gl_FragColor = texture2D(uTex, uRect.xy + uv * uRect.zw);
}`;

/** v_fov from h_fov and the output aspect (rectilinear), unless pinned --
 *  the same rule vr_project.py's auto_v_fov applies before it calls v360. */
function autoVFov(hFov, outW, outH) {
  const h = hFov * D2R;
  return Math.atan(Math.tan(h / 2) * outH / outW) * 2 / D2R;
}

function vFovOf(cfg) {
  return cfg.v_fov || autoVFov(cfg.h_fov, cfg.out_w, cfg.out_h);
}

/** Where one eye sits in the source frame, as a normalized rect. */
function eyeRect(layout) {
  if (layout === "sbs") return [0, 0, 0.5, 1];
  if (layout === "tb") return [0, 0, 1, 0.5];
  return [0, 0, 1, 1];
}

/** Ry(yaw) * Rp(pitch) * Rr(roll), returned column-major for WebGL. */
function rotation(yaw, pitch, roll) {
  const cy = Math.cos(yaw * D2R), sy = Math.sin(yaw * D2R);
  const cp = Math.cos(pitch * D2R), sp = Math.sin(pitch * D2R);
  const cr = Math.cos(roll * D2R), sr = Math.sin(roll * D2R);
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const Rp = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]];
  const Rr = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]];
  const mul = (A, B) => A.map((row, i) =>
    [0, 1, 2].map(j => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]));
  const M = mul(mul(Ry, Rp), Rr);
  const out = new Float32Array(9);           // column-major for uniformMatrix3fv
  for (let c = 0; c < 3; c++)
    for (let r = 0; r < 3; r++) out[c * 3 + r] = M[r][c];
  return out;
}

/** Fraction of the viewport that falls outside the source's coverage (a
 *  180 deg eye or a fisheye lens circle). > 0 means the render will smear
 *  edge texels there. Mirrors the shader on a coarse grid -- this is a
 *  warning, not a measurement. */
function coverage(cfg, yaw, pitch, nx, ny) {
  nx = nx || 41; ny = ny || 23;
  if (cfg.projection === "equirect") return 0;
  const R = rotation(yaw, pitch, cfg.roll || 0);
  const tx = Math.tan(cfg.h_fov * D2R / 2), ty = Math.tan(vFovOf(cfg) * D2R / 2);
  const lensHalf = (cfg.ih_fov || 180) * D2R / 2;
  let out = 0;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const x = (2 * i + 1) / nx - 1, y = (2 * j + 1) / ny - 1;
      let vx = x * tx, vy = y * ty, vz = 1;
      const n = Math.hypot(vx, vy, vz);
      vx /= n; vy /= n; vz /= n;
      // column-major: R[c*3+r]
      const X = R[0] * vx + R[3] * vy + R[6] * vz;
      const Y = R[1] * vx + R[4] * vy + R[7] * vz;
      const Z = R[2] * vx + R[5] * vy + R[8] * vz;
      if (cfg.projection === "fisheye") {
        if (Math.acos(Math.max(-1, Math.min(1, Z))) > lensHalf) out++;
      } else if (Math.abs(Math.atan2(X, Z)) > Math.PI / 2) out++;
    }
  }
  return out / (nx * ny);
}

/** A WebGL viewport projector bound to one canvas. `ok` is false when the
 *  browser gave us no GL context -- callers fall back to server rendering. */
class Projector {
  constructor(canvas) {
    this.canvas = canvas;
    this.ok = false;
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false })
            || canvas.getContext("webgl", { alpha: false, antialias: false });
    if (!gl) return;
    this.gl = gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(s) || "shader compile failed");
      return s;
    };
    try {
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
      gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(p) || "program link failed");
      this.prog = p;
    } catch (e) {
      this.error = String(e);
      return;
    }
    gl.useProgram(this.prog);
    this.u = {};
    for (const k of ["uTex", "uTan", "uRot", "uInput", "uLensHalf", "uRect"])
      this.u[k] = gl.getUniformLocation(this.prog, k);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER,
                  new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    // CLAMP_TO_EDGE is load-bearing: it is what makes an over-aimed viewport
    // smear exactly the way v360 smears it in the render.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);   // row 0 = image top
    this.hasSource = false;
    this.ok = true;
  }

  /** Upload a frame (HTMLImageElement / HTMLVideoElement / ImageBitmap /
   *  canvas). Cheap enough to call per animation frame for video. */
  upload(src) {
    if (!this.ok) return false;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, src);
    this.hasSource = true;
    return true;
  }

  /** Project the uploaded frame at (yaw, pitch). `rect` optionally selects the
   *  eye inside the texture -- omit it when the server already cropped. */
  draw(cfg, yaw, pitch, rect) {
    if (!this.ok || !this.hasSource) return false;
    const gl = this.gl, c = this.canvas;
    const w = c.width, h = c.height;
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    gl.uniform2f(this.u.uTan,
                 Math.tan(cfg.h_fov * D2R / 2),
                 Math.tan(vFovOf(cfg) * D2R / 2));
    gl.uniformMatrix3fv(this.u.uRot, false,
                        rotation(yaw, pitch, cfg.roll || 0));
    gl.uniform1i(this.u.uInput, PROJ[cfg.projection] ?? 1);
    gl.uniform1f(this.u.uLensHalf, (cfg.ih_fov || 180) * D2R / 2);
    const r = rect || [0, 0, 1, 1];
    gl.uniform4f(this.u.uRect, r[0], r[1], r[2], r[3]);
    gl.uniform1i(this.u.uTex, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return true;
  }

  /** Size the drawing buffer to the aim's output aspect, capped by CSS width. */
  resize(cssW, cfg) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(2, Math.round(cssW * dpr));
    const h = Math.max(2, Math.round(w * cfg.out_h / cfg.out_w));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }
}

root.VRProjector = { Projector, autoVFov, vFovOf, eyeRect, rotation, coverage };

})(typeof window !== "undefined" ? window : globalThis);
