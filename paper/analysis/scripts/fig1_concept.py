"""Fig. 1 | One optical projection links per-exposure fitting, feed-forward
reconstruction and velocimetry.

Schematic-led composite. The tracer field, sub-aperture image and linked
displacements are archived simulation data; the optics, footprints and block
diagrams are drawings.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

from lfstyle import C, DATA, MM, ax_mm, image_axes, new_figure, save

H = 110.0
GOLD = C["gold"]


def box(ax, x, y, w, h, text, fc="white", ec=C["ink"], color=C["ink"], fs=6.2, lw=0.7, bold=False, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, zorder=4,
            linespacing=1.15, fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color=C["ink"], lw=0.8, style="-|>", rad=0.0, ls="-", z=5):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=6,
                                connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def blob(ax, x, y, r, color="white"):
    for k, a in ((1.0, 0.18), (0.72, 0.35), (0.45, 0.65), (0.22, 1.0)):
        ax.add_patch(Circle((x, y), r * k, fc=color, ec="none", alpha=a, zorder=6))


def main():
    fig = new_figure(H)
    s = fig.add_axes([0, 0, 1, 1])
    s.set_xlim(0, 180)
    s.set_ylim(H, 0)          # mm from top-left, y downwards
    s.axis("off")

    def label(letter, x, y):
        s.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="top")

    def head(text, x, y):
        s.text(x, y, text, fontsize=7, fontweight="bold", ha="left", va="top")

    # =================================================================== a
    label("a", 2, 2)
    head("Light-field measurement", 7, 2.3)
    field = np.load(DATA / "fig2/fields/sample_00000_diam50um_conc0p025/figure_data.npz")
    gt = field["gt"]
    oy, src_x = 24.0, 39.2
    ax3 = fig.add_axes([2 / 180, 1 - 44 / H, 38 / 180, 38 / H], projection="3d")
    ax3.scatter(gt[:, 0], gt[:, 2], gt[:, 1], s=0.9, color=C["mid"], alpha=0.55, linewidths=0, depthshade=False)
    hi = np.array([2.0, 0.5, 1.6])
    ax3.scatter([hi[0]], [hi[2]], [hi[1]], s=14, color=GOLD, linewidths=0, depthshade=False, zorder=10)
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-5, 5)
    ax3.set_zlim(-8, 8)
    ax3.set_box_aspect((20, 10, 16))
    ax3.view_init(elev=18, azim=-62)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_ticks([])
        axis.pane.fill = False
        axis.pane.set_edgecolor(C["light"])
        axis.line.set_color(C["light"])
    ax3.grid(False)
    ax3.patch.set_alpha(0)
    s.text(21, 41.3, "1,078 tracers in 20 × 16 × 10 mm", fontsize=5.8, ha="center", va="top", color=C["mid"])
    s.plot([27.5, src_x - 0.9], [22.6, oy - 1.5], color=GOLD, lw=0.5, ls=(0, (1.5, 1.5)), zorder=2)

    # optics sketch
    s.add_patch(Ellipse((47, oy), 2.6, 21, fc="#DCE9F5", ec=C["net"], lw=0.7, zorder=3))
    for k in range(9):
        yy = oy - 10 + k * 2.5
        s.add_patch(Ellipse((63, yy), 1.1, 2.3, fc="#DCE9F5", ec=C["net"], lw=0.5, zorder=3))
    s.add_patch(Rectangle((66.3, oy - 11.2), 1.0, 22.4, fc=C["ink"], ec="none", zorder=3))
    src = (src_x, oy - 1.5)
    for yl, col in ((oy - 8.5, "#E0A33A"), (oy, GOLD), (oy + 8.5, "#9C6A12")):
        s.plot([src[0], 47], [src[1], yl], color=col, lw=0.6, zorder=2)
        s.plot([47, 63], [yl, oy + 1.25], color=col, lw=0.6, zorder=2)
    for yt, col in ((oy + 2.3, "#E0A33A"), (oy + 1.25, GOLD), (oy + 0.2, "#9C6A12")):
        s.plot([63, 66.3], [oy + 1.25, yt], color=col, lw=0.6, zorder=2)
    s.add_patch(Circle(src, 0.7, fc=GOLD, ec="none", zorder=4))
    s.text(47, 37.3, "Main lens", fontsize=5.8, ha="center", va="top")
    s.text(64.5, 37.3, "Microlenses\nand sensor", fontsize=5.8, ha="center", va="top", linespacing=1.1)
    s.text(55, 11.2, "Each sensor pixel behind a\nmicrolens sees one direction", fontsize=5.5, ha="center",
           va="bottom", color=C["mid"], linespacing=1.1)

    # stack of sub-aperture images
    sai = field["central_sai"][53:133, 66:166] / 255.0
    for k, (dx, dy) in enumerate(((5.0, -5.0), (2.5, -2.5), (0.0, 0.0))):
        axi = ax_mm(fig, 72 + dx, 15.5 + dy, 21, 16.8)
        axi.imshow(sai if k == 2 else np.roll(sai, (k + 1) * 2, axis=1), cmap="gray", vmin=0,
                   vmax=np.percentile(sai, 99.7), interpolation="nearest")
        image_axes(axi)
        for sp in axi.spines.values():
            sp.set_visible(True)
            sp.set_color("white")
            sp.set_linewidth(0.6)
    arrow(s, (68.5, oy), (71.5, oy))
    s.text(85, 37.3, "22 × 22 sub-aperture\nimages (SAIs)", fontsize=5.8, ha="center", va="top", linespacing=1.1)

    # =================================================================== b
    bx = 104.0
    label("b", bx - 3, 2)
    head("A Gaussian particle and its projection", bx + 2, 2.3)
    s.text(bx + 2, 9.6, "Particle i", fontsize=6.4, fontweight="bold", ha="left", va="center")
    rows = [(r"$\mathbf{X}_i$", "centre (x, y, z)"), (r"$A_i$", "strength"), (r"$\sigma_i$", "apparent width")]
    for k, (sym, txt) in enumerate(rows):
        s.text(bx + 3.5, 14.4 + k * 4.2, sym, fontsize=7.4, ha="left", va="center")
        s.text(bx + 9.0, 14.4 + k * 4.2, txt, fontsize=6.0, ha="left", va="center")
    # views with projected footprints
    vx0, vy0, vs, vg = bx + 34.0, 9.6, 9.2, 1.6
    shifts = (-2.3, -0.8, 0.8, 2.3)
    for k, sh in enumerate(shifts):
        x = vx0 + k * (vs + vg)
        s.add_patch(Rectangle((x, vy0), vs, vs, fc="#1E1E1E", ec="none", zorder=3))
        blob(s, x + vs / 2 + sh, vy0 + vs / 2 - 0.35 * sh, 1.9)
        s.add_patch(Circle((x + vs / 2 + sh, vy0 + vs / 2 - 0.35 * sh), 0.35, fc=GOLD, ec="none", zorder=7))
    s.text(vx0 + 2 * vs + 1.5 * vg, vy0 - 0.6, r"Four of the 484 views $a$", fontsize=5.8, ha="center",
           va="bottom", color=C["mid"])
    s.text(vx0 + 2 * vs + 1.5 * vg, vy0 + vs + 1.0, r"footprint centred at $\pi_a(\mathbf{X}_i)$", fontsize=7.2,
           ha="center", va="top")
    arrow(s, (bx + 25.5, 18.6), (vx0 - 1.2, 14.6), color=GOLD, lw=0.9)
    s.text(bx + 27.0, 14.6, r"$\pi_a$", fontsize=7.5, ha="center", va="center", color=GOLD)
    # rendering equation
    s.add_patch(FancyBboxPatch((bx + 1.5, 29.4), 72.5, 13.0, boxstyle="round,pad=0,rounding_size=1.2",
                               fc="#FBF4E6", ec=GOLD, lw=0.7, zorder=2))
    s.text(bx + 37.75, 33.6,
           r"$\hat{I}_a(\mathbf{q}) \;=\; \sum_i A_i \exp\!\left(-\,\|\mathbf{q}-\pi_a(\mathbf{X}_i)\|^{2}\,/\,2\sigma_i^{2}\right)$",
           fontsize=7.6, ha="center", va="center", zorder=4)
    s.text(bx + 37.75, 39.2, r"Optical projection $\pi_a$ is fixed by the camera calibration",
           fontsize=7.2, ha="center", va="center", color="#7A5410", zorder=4)

    # =================================================================== c
    label("c", 2, 50)
    head("Two routes to the same particles", 7, 50.3)
    # input images
    for k, (dx, dy) in enumerate(((1.6, -1.6), (0.8, -0.8), (0.0, 0.0))):
        s.add_patch(Rectangle((5 + dx, 75.6 + dy), 9.5, 7.6, fc="#1E1E1E", ec="white", lw=0.5, zorder=3 + k))
    for px, py in ((7.2, 77.8), (10.6, 81.0), (12.4, 77.3), (8.6, 81.8)):
        s.add_patch(Circle((px, py), 0.3, fc="white", ec="none", zorder=7))
    s.text(10.5, 85.0, "SAIs", fontsize=6.0, ha="center", va="top")

    # lane 1: per-exposure fitting
    fitc, netc = C["fit"], C["net"]
    y1, hb = 57.5, 9.5
    s.add_patch(FancyBboxPatch((20, y1 - 2.2), 99, hb + 9.3, boxstyle="round,pad=0,rounding_size=1.5",
                               fc="#EEF7F5", ec="none", zorder=1))
    s.text(22, y1 - 0.4, "Per-exposure fitting", fontsize=6.4, fontweight="bold", color=fitc, ha="left", va="top")
    s.text(117, y1 - 0.4, "2,000 steps, ≈37 s", fontsize=6.0, color=fitc, ha="right", va="top")
    bx1 = [(22, 23, "Candidate particles\nfrom backprojection"), (51, 20, "Render\n32 views"),
           (77, 20, "Compare with\nobserved SAIs")]
    for x, w, t in bx1:
        box(s, x, y1 + 3.3, w, hb - 1.3, t, ec=fitc)
    arrow(s, (45, y1 + 7.4), (51, y1 + 7.4), color=fitc)
    arrow(s, (71, y1 + 7.4), (77, y1 + 7.4), color=fitc)
    # gradient loop back
    s.plot([93, 93], [y1 + 11.5, y1 + 15.2], color=fitc, lw=0.8, zorder=5)
    s.plot([93, 62.2], [y1 + 15.2, y1 + 15.2], color=fitc, lw=0.8, zorder=5)
    s.plot([59.8, 33.5], [y1 + 15.2, y1 + 15.2], color=fitc, lw=0.8, zorder=5)
    arrow(s, (33.5, y1 + 15.2), (33.5, y1 + 11.5), color=fitc)
    s.text(77.6, y1 + 14.3, r"gradient with respect to $\mathbf{X}, A, \sigma$", fontsize=5.6, color=fitc,
           ha="center", va="bottom", zorder=6)

    # shared projection bar
    yb = 78.6
    box(s, 31, yb, 72, 5.4, r"Shared optical projection $\pi_a(\mathbf{X})$  ·  camera calibration", fc="#FBF4E6",
        ec=GOLD, color="#7A5410", fs=7.2, bold=True)
    arrow(s, (61, yb), (61, y1 + 11.5), color=GOLD, lw=0.9)

    # lane 2: feed-forward network
    y2 = 88.4
    arrow(s, (61, yb + 5.4), (61, y2 + 2.3), color=GOLD, lw=0.9)
    s.add_patch(FancyBboxPatch((20, y2 - 1.2), 99, hb + 7.2, boxstyle="round,pad=0,rounding_size=1.5",
                               fc="#EDF3FA", ec="none", zorder=1))
    s.text(22, y2 + hb + 5.4, "Feed-forward network", fontsize=6.4, fontweight="bold", color=netc, ha="left",
           va="bottom")
    s.text(117, y2 + hb + 5.4, "one pass, 158 ms", fontsize=6.0, color=netc, ha="right", va="bottom")
    bx2 = [(22, 19, "2D encoder\n32 views"), (46, 30, "Sample features at\n" + r"$\pi_a(\mathbf{X})$, 64 depths"),
           (81, 17, "Mean and\nvariance"), (101, 16, "3D\ndecoder")]
    for x, w, t in bx2:
        box(s, x, y2 + 2.3, w, hb - 1.3, t, ec=netc, fs=7.2 if "pi_a" in t else 6.2)
    arrow(s, (41, y2 + 6.4), (46, y2 + 6.4), color=netc)
    arrow(s, (76, y2 + 6.4), (81, y2 + 6.4), color=netc)
    arrow(s, (98, y2 + 6.4), (101, y2 + 6.4), color=netc)
    # input arrows
    s.plot([16.8, 18.4], [79.4, 79.4], color=C["ink"], lw=0.8, zorder=5)
    s.plot([18.4, 18.4], [y1 + 7.4, y2 + 6.4], color=C["ink"], lw=0.8, zorder=5)
    arrow(s, (18.4, y1 + 7.4), (22, y1 + 7.4))
    arrow(s, (18.4, y2 + 6.4), (22, y2 + 6.4))

    # common output
    box(s, 123.5, 72.0, 14.5, 18.5, "Gaussian\nparticles\n" + r"$\{\mathbf{X}_i, A_i, \sigma_i\}$", fc="white",
        ec=C["ink"], fs=7.2, bold=False, lw=0.9)
    s.plot([97, 121], [y1 + 7.4, y1 + 7.4], color=fitc, lw=0.8, zorder=5)
    s.plot([121, 121], [y1 + 7.4, 76.5], color=fitc, lw=0.8, zorder=5)
    arrow(s, (121, 76.5), (123.5, 76.5), color=fitc)
    s.plot([117, 121], [y2 + 6.4, y2 + 6.4], color=netc, lw=0.8, zorder=5)
    s.plot([121, 121], [y2 + 6.4, 86.0], color=netc, lw=0.8, zorder=5)
    arrow(s, (121, 86.0), (123.5, 86.0), color=netc)

    # =================================================================== d
    label("d", 141, 50)
    head("Two-frame displacement", 146, 50.3)
    links = np.load(DATA / "fig1/example_motion_links.npz")
    sel = links["xyz0_in_central_cube"]
    p0, p1, ok = links["endpoint0_mm"][sel], links["endpoint1_mm"][sel], links["correct"][sel]
    axd = ax_mm(fig, 152, 62.5, 24, 30)
    for a0, a1, good in zip(p0, p1, ok):
        col = C["net"] if good else C["bad"]
        axd.annotate("", xy=(a1[0], a1[2]), xytext=(a0[0], a0[2]),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=5))
        axd.scatter([a0[0]], [a0[2]], s=7, facecolors="white", edgecolors=col, linewidths=0.6, zorder=4)
    axd.set_xlim(-2.2, 2.2)
    axd.set_ylim(-2.5, 2.5)
    axd.set_xticks([-2, 0, 2])
    axd.set_yticks([-2, 0, 2])
    axd.set_xlabel("x (mm)")
    axd.set_ylabel("Depth z (mm)", labelpad=1)
    axd.tick_params(labelsize=5.8)
    s.text(164, 102.5, "Linked centres, +0.20-mm axial step", fontsize=5.6, ha="center", va="top", color=C["mid"])
    arrow(s, (138, 81.25), (143.5, 81.25))

    save(fig, "fig1_concept")
    print("links shown", int(sel.sum()), "correct", int(ok.sum()))


if __name__ == "__main__":
    main()
