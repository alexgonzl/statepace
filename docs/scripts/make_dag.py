"""Render theoretical_framework_dag.{png,svg} from parameters using matplotlib."""
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# ---- parameters ------------------------------------------------------------
NODE_DIAM     = 1.0
MIN_EDGE_LEN  = 1.1 * NODE_DIAM
STEP          = NODE_DIAM + MIN_EDGE_LEN
R             = NODE_DIAM / 2

FILL_COLOR    = "#e8e8e8"
STROKE_COLOR  = "#333333"
NODE_LINEW    = 1.4
EDGE_LINEW    = 1.4
FONTSIZE      = 16
MUTATION      = 18                # matplotlib arrowhead scale
CURVE_RAD     = -0.25              # arc3 curvature; + = left of chord, - = right
STUB_LEN      = 0.6 * NODE_DIAM    # length of dotted continuity stubs
STUB_LINEW    = 1.2
STUB_DASH     = (0, (3, 3))        # dotted/dashed pattern

X_ZPREV, X_P, X_X, X_Z = 0, STEP, 2*STEP, 3*STEP
Y_TOP, Y_MID, Y_BOT    = 2*STEP, STEP, 0

NODES = {
    "Zprev": dict(label=r"$Z_{t-1}$", x=X_ZPREV, y=Y_TOP, filled=False),
    "Z":     dict(label=r"$Z_{t}$",   x=X_Z,     y=Y_TOP, filled=False),
    "P":     dict(label=r"$P_{t}$",   x=X_P,     y=Y_MID, filled=True),
    "X":     dict(label=r"$X_{t}$",   x=X_X,     y=Y_MID, filled=True),
    "E":     dict(label=r"$E_{t}$",   x=X_X,     y=Y_BOT, filled=True),
}

# edges: (src, dst, rad)  rad=0 is straight; sign controls bow direction
EDGES = [
    ("Zprev", "Z",  0.0),
    ("Zprev", "X",  0.0),
    ("P",     "X",  0.0),
    ("E",     "X",  0.0),
    ("Zprev", "P", -CURVE_RAD),   # bows down-left (right of chord as we go Zprev->P)
    ("X",     "Z", -CURVE_RAD),   # bows up-right
]

def _draw_node(ax, spec):
    face = FILL_COLOR if spec["filled"] else "white"
    ax.add_patch(Circle((spec["x"], spec["y"]), R,
                        facecolor=face, edgecolor=STROKE_COLOR,
                        linewidth=NODE_LINEW, zorder=2))
    ax.text(spec["x"], spec["y"], spec["label"],
            ha="center", va="center", fontsize=FONTSIZE, zorder=3)

def _draw_edge(ax, src, dst, rad):
    p0 = (NODES[src]["x"], NODES[src]["y"])
    p1 = (NODES[dst]["x"], NODES[dst]["y"])
    # shrinkA/shrinkB in points: convert radius (data units) to points via fig dpi.
    # Simpler: use axes coords by converting R from data->points at current transform.
    # matplotlib's shrinkA/shrinkB are in points; we pass R converted.
    arrow = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=MUTATION,
        color=STROKE_COLOR, linewidth=EDGE_LINEW,
        shrinkA=0, shrinkB=0, zorder=1,
        patchA=None, patchB=None,   # we'll set per-call below
    )
    ax.add_patch(arrow)
    return arrow

def render(out_stem):
    margin = 0.4
    w_data = 3*STEP + NODE_DIAM
    h_data = 2*STEP + NODE_DIAM
    fig, ax = plt.subplots(figsize=(w_data + 2*margin, h_data + 2*margin))
    ax.set_xlim(-R - STUB_LEN - margin, 3*STEP + R + STUB_LEN + margin)
    ax.set_ylim(-R - margin, 2*STEP + R + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    # draw nodes first, collect circle patches keyed by node name
    circles = {}
    for name, spec in NODES.items():
        face = FILL_COLOR if spec["filled"] else "white"
        c = Circle((spec["x"], spec["y"]), R,
                   facecolor=face, edgecolor=STROKE_COLOR,
                   linewidth=NODE_LINEW, zorder=2)
        ax.add_patch(c)
        circles[name] = c
        ax.text(spec["x"], spec["y"], spec["label"],
                ha="center", va="center", fontsize=FONTSIZE, zorder=3)

    # draw edges with patchA/patchB = node circles → matplotlib clips at boundary
    for src, dst, rad in EDGES:
        arrow = FancyArrowPatch(
            (NODES[src]["x"], NODES[src]["y"]),
            (NODES[dst]["x"], NODES[dst]["y"]),
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=MUTATION,
            color=STROKE_COLOR, linewidth=EDGE_LINEW,
            shrinkA=0, shrinkB=0, zorder=1,
            patchA=circles[src], patchB=circles[dst],
        )
        ax.add_patch(arrow)

    # continuity stubs: dotted arrows into Z_{t-1} (from left) and out of Z_t (to right)
    zprev = NODES["Zprev"]
    z     = NODES["Z"]
    # stub into Zprev: dashed line + solid arrowhead at the node boundary
    stub_start_x = zprev["x"] - STUB_LEN - R
    stub_end_x   = zprev["x"] - R            # lands at Zprev's west boundary
    # dashed segment (no arrowhead) — stops short of boundary to leave room for arrow tip
    head_room = 0.18 * NODE_DIAM
    ax.plot([stub_start_x, stub_end_x - head_room], [zprev["y"], zprev["y"]],
            color=STROKE_COLOR, linewidth=STUB_LINEW, linestyle=STUB_DASH,
            zorder=1, solid_capstyle="butt")
    # solid arrowhead tip into Zprev
    ax.add_patch(FancyArrowPatch(
        (stub_end_x - head_room, zprev["y"]),
        (zprev["x"], zprev["y"]),
        arrowstyle="-|>", mutation_scale=MUTATION,
        color=STROKE_COLOR, linewidth=STUB_LINEW,
        shrinkA=0, shrinkB=0, zorder=1,
        patchB=circles["Zprev"],
    ))

    # dashed tail leaving Z_t eastward (outgoing continuity, no arrowhead)
    ax.plot([z["x"] + R, z["x"] + R + STUB_LEN], [z["y"], z["y"]],
            color=STROKE_COLOR, linewidth=STUB_LINEW, linestyle=STUB_DASH,
            zorder=1, solid_capstyle="butt")

    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight",
                    pad_inches=0.1, facecolor="white")
    plt.close(fig)

def main():
    docs_dir = Path(__file__).parent.parent   # docs/
    out = docs_dir / "theoretical_framework_dag"
    render(str(out))
    print(f"wrote {out}.png, {out}.svg")

if __name__ == "__main__":
    main()
