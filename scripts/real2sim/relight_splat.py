"""Put the scene splat into the deployment camera's photometric space.

The splat was trained on phone photos, which expose the black tablecloth far darker than
the ZED does: measured over the same surfaces the ZED is brighter by a gain of about
1.35 to 1.53 per channel. Rendered next to a real frame our scene therefore comes out
44 percent too dark, which is a domain gap the policy would see.

Correcting the composed image instead would also brighten the raytraced robot and objects,
which already match, so the correction belongs to the splat. Spherical harmonics are
linear in radiance, so a gain and offset on the output colour is exactly a gain on every
coefficient plus an offset on the DC term:

    colour = 0.28209479 * f_dc + 0.5          (3DGS/2DGS convention)
    colour' = g * colour + o
    f_dc'   = g * f_dc + (o + 0.5 * g - 0.5) / 0.28209479
    f_rest' = g * f_rest

No retraining, and the geometry is untouched.

With `--region` the same correction applies to a world-space box only - a local relight.
Used for the table's front face, whose baked edge glow the global relight (fitted on the
tabletop and curtains) overshoots by ~25 percent against the real frame.
"""

import argparse

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splat", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gain", type=float, nargs=3, required=True, help="per channel, R G B")
    p.add_argument("--offset", type=float, nargs=3, required=True, help="per channel, in 0-255")
    p.add_argument("--region", type=float, nargs=6, default=None,
                   metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                   help="apply to gaussians inside this world-space box only")
    args = p.parse_args()

    ply = PlyData.read(args.splat)
    vert = ply["vertex"]
    data = vert.data.copy()
    names = data.dtype.names

    sel = np.ones(len(data), bool)
    if args.region is not None:
        x0, x1, y0, y1, z0, z1 = args.region
        xyz = np.stack([data["x"], data["y"], data["z"]], 1)
        sel = ((xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1)
               & (xyz[:, 2] > z0) & (xyz[:, 2] < z1))
        print(f"region box: {int(sel.sum())} of {len(data)} gaussians")

    gain = np.array(args.gain, dtype=np.float64)
    offset = np.array(args.offset, dtype=np.float64) / 255.0
    print(f"gain {gain}, offset {offset * 255} of 255")

    for c in range(3):
        key = f"f_dc_{c}"
        if key not in names:
            raise SystemExit(f"{key} missing; is this a gaussian splat ply?")
        v = data[key].astype(np.float64)
        v[sel] = gain[c] * v[sel] + (offset[c] + 0.5 * gain[c] - 0.5) / C0
        data[key] = v.astype(data[key].dtype)

    rest = [n for n in names if n.startswith("f_rest_")]
    if rest:
        # the rest coefficients are stored channel-major: 15 for red, then green, then blue
        per = len(rest) // 3
        for c in range(3):
            for k in range(per):
                key = f"f_rest_{c * per + k}"
                v = data[key].astype(np.float64)
                v[sel] = gain[c] * v[sel]
                data[key] = v.astype(data[key].dtype)
        print(f"scaled {len(rest)} higher-order coefficients")

    PlyData([PlyElement.describe(data, "vertex")]).write(args.out)
    print(f"wrote {args.out} with {len(data)} gaussians")


if __name__ == "__main__":
    main()
