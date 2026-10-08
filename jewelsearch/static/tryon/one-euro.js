// One-Euro filter (Casiez et al. 2012): heavy smoothing when the hand is
// still (kills landmark jitter), light smoothing when it moves (no lag).

const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));

class OneEuro {
  constructor(minCutoff, beta, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0 });
  }
  filter(value, dt) {
    if (this.x === null) { this.x = value; return value; }
    const dx = (value - this.x) / dt;
    this.dx += alpha(this.dCutoff, dt) * (dx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += alpha(cutoff, dt) * (value - this.x);
    return this.x;
  }
}

// Filters every coordinate of a fixed-length list of {x, y, z} points.
export class PointsFilter {
  constructor(count, { minCutoff = 1.2, beta = 2.5 } = {}) {
    this.f = Array.from({ length: count * 3 }, () => new OneEuro(minCutoff, beta));
    this.t = null;
  }
  reset() { this.f.forEach((f) => { f.x = null; f.dx = 0; }); this.t = null; }
  apply(points, timeMs) {
    const dt = this.t === null ? 1 / 30 : Math.max(1e-3, (timeMs - this.t) / 1000);
    this.t = timeMs;
    return points.map((p, i) => ({
      x: this.f[i * 3].filter(p.x, dt),
      y: this.f[i * 3 + 1].filter(p.y, dt),
      z: this.f[i * 3 + 2].filter(p.z, dt),
    }));
  }
}
