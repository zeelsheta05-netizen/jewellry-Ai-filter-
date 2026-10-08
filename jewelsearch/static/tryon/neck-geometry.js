// Where a chain sits on a neck photo, from the chin, the shoulder joints and
// the real-size scale (px per mm). All points are normalised 0..1.
//
// The pose model's shoulder points are the shoulder joints, which sit ~45 mm
// below the line where the neck meets the shoulders, on every body and every
// camera distance (checked on close and far photos). Measuring down from the
// chin instead depends on posture and how far the head is tilted.

const BASE_ABOVE_SHOULDER_MM = 45;
const NOTCH_BELOW_BASE_MM = 18;   // the hollow at the base of the throat

export const NECK_VERSION = 2;

export function neckAnchors({ chin, shoulderL, shoulderR, halfWidth, pxPerMm, H }) {
  const mmY = pxPerMm / H;
  const shoulderY = (shoulderL.y + shoulderR.y) / 2;
  const shoulderX = (shoulderL.x + shoulderR.x) / 2;
  const baseY = Math.max(chin.y + 8 * mmY, shoulderY - BASE_ABOVE_SHOULDER_MM * mmY);
  return {
    neckL: { x: chin.x - halfWidth, y: baseY },
    neckR: { x: chin.x + halfWidth, y: baseY },
    notch: { x: (chin.x + shoulderX) / 2, y: baseY + NOTCH_BELOW_BASE_MM * mmY },
    v: NECK_VERSION,
  };
}
