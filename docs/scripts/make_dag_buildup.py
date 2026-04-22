"""Render a build-up sequence of the workout-day DAG.

Produces ``docs/figures/dag_buildup/step{1..6}.{png,svg}``. Each step
introduces one new node or edge along the causal story:

  1. Z_{t-1} — latent state alone
  2. + P_t and Z_{t-1} -> P_t  (selection)
  3. + E_t                      (exogenous; no edges yet)
  4. + X_t with Z_{t-1} -> X_t, P_t -> X_t, E_t -> X_t  (observation)
  5. + Z_t and X_t -> Z_t       (intervention)
  6. + Z_{t-1} -> Z_t and continuity stubs  (full DAG)

Layout, colors, and arrow style mirror ``make_dag.py`` so the build-up and the
final figure are visually consistent.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# ---- parameters (kept in sync with make_dag.py) ----------------------------
NODE_DIAM    = 1.0
MIN_EDGE_LEN = 1.1 * NODE_DIAM
STEP         = NODE_DIAM + MIN_EDGE_LEN
R            = NODE_DIAM / 2

FILL_COLOR   = "#e8e8e8"
STROKE_COLOR = "#333333"
MUTED_COLOR  = "#cccccc"   # for nodes/edges not yet introduced (kept hidden, not drawn)
NEW_COLOR    = "#c0392b"   # highlight color for newly added node/edge
NODE_LINEW   = 1.4
EDGE_LINEW   = 1.4
NEW_LINEW    = 2.2
FONTSIZE     = 16
MUTATION     = 18
CURVE_RAD    = -0.25
STUB_LEN     = 0.6 * NODE_DIAM
STUB_LINEW   = 1.2
STUB_DASH    = (0, (3, 3))

X_ZPREV, X_P, X_X, X_Z = 0, STEP, 2 * STEP, 3 * STEP
Y_TOP, Y_MID, Y_BOT    = 2 * STEP, STEP, 0

NODES = {
    "Zprev": dict(label=r"$Z_{t-1}$", x=X_ZPREV, y=Y_TOP, filled=False),
    "Z":     dict(label=r"$Z_{t}$",   x=X_Z,     y=Y_TOP, filled=False),
    "P":     dict(label=r"$P_{t}$",   x=X_P,     y=Y_MID, filled=True),
    "X":     dict(label=r"$X_{t}$",   x=X_X,     y=Y_MID, filled=True),
    "E":     dict(label=r"$E_{t}$",   x=X_X,     y=Y_BOT, filled=True),
}

# All edges in the full DAG, with arc curvature.
ALL_EDGES = {
    ("Zprev", "P"): -CURVE_RAD,   # selection (bowed)
    ("Zprev", "X"): 0.0,          # direct capacity
    ("P",     "X"): 0.0,          # session shape -> execution
    ("E",     "X"): 0.0,          # exogenous -> execution
    ("X",     "Z"): -CURVE_RAD,   # intervention (bowed)
    ("Zprev", "Z"): 0.0,          # autonomous dynamics
}


# ---- step specifications ---------------------------------------------------

def _step_specs():
    """Return list of (title, visible_nodes_set, visible_edges_set, new_nodes, new_edges)."""
    steps = []

    # Step 1 — Z_{t-1} alone
    steps.append(dict(
        title="Step 1 — latent state $Z_{t-1}$",
        nodes={"Zprev"},
        edges=set(),
        new_nodes={"Zprev"},
        new_edges=set(),
        stubs=False,
    ))

    # Step 2 — + P_t, selection edge
    steps.append(dict(
        title="Step 2 — session selection: $Z_{t-1} \\to P_t$",
        nodes={"Zprev", "P"},
        edges={("Zprev", "P")},
        new_nodes={"P"},
        new_edges={("Zprev", "P")},
        stubs=False,
    ))

    # Step 3 — + E_t (no edges)
    steps.append(dict(
        title="Step 3 — exogenous environment $E_t$",
        nodes={"Zprev", "P", "E"},
        edges={("Zprev", "P")},
        new_nodes={"E"},
        new_edges=set(),
        stubs=False,
    ))

    # Step 4 — + X_t with observation edges
    steps.append(dict(
        title="Step 4 — execution $X_t$: observation edges",
        nodes={"Zprev", "P", "E", "X"},
        edges={("Zprev", "P"), ("Zprev", "X"), ("P", "X"), ("E", "X")},
        new_nodes={"X"},
        new_edges={("Zprev", "X"), ("P", "X"), ("E", "X")},
        stubs=False,
    ))

    # Step 5 — + Z_t and X -> Z (intervention)
    steps.append(dict(
        title="Step 5 — intervention: $X_t \\to Z_t$",
        nodes={"Zprev", "P", "E", "X", "Z"},
        edges={("Zprev", "P"), ("Zprev", "X"), ("P", "X"), ("E", "X"),
               ("X", "Z")},
        new_nodes={"Z"},
        new_edges={("X", "Z")},
        stubs=False,
    ))

    # Step 6 — + Z_{t-1} -> Z_t and continuity stubs (full DAG)
    steps.append(dict(
        title="Step 6 — autonomous dynamics: $Z_{t-1} \\to Z_t$ + continuity",
        nodes={"Zprev", "P", "E", "X", "Z"},
        edges=set(ALL_EDGES.keys()),
        new_nodes=set(),
        new_edges={("Zprev", "Z")},
        stubs=True,
    ))

    return steps


# ---- rendering -------------------------------------------------------------

def _make_axes():
    margin = 0.4
    w_data = 3 * STEP + NODE_DIAM
    h_data = 2 * STEP + NODE_DIAM
    fig, ax = plt.subplots(figsize=(w_data + 2 * margin, h_data + 2 * margin + 0.4))
    ax.set_xlim(-R - STUB_LEN - margin, 3 * STEP + R + STUB_LEN + margin)
    ax.set_ylim(-R - margin, 2 * STEP + R + margin + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _draw_node(ax, name, *, highlight=False):
    spec = NODES[name]
    face = FILL_COLOR if spec["filled"] else "white"
    edge = NEW_COLOR if highlight else STROKE_COLOR
    lw   = NEW_LINEW if highlight else NODE_LINEW
    c = Circle((spec["x"], spec["y"]), R,
               facecolor=face, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(c)
    ax.text(spec["x"], spec["y"], spec["label"],
            ha="center", va="center", fontsize=FONTSIZE, zorder=3)
    return c


def _draw_edge(ax, src, dst, rad, circles, *, highlight=False):
    color = NEW_COLOR if highlight else STROKE_COLOR
    lw    = NEW_LINEW if highlight else EDGE_LINEW
    arrow = FancyArrowPatch(
        (NODES[src]["x"], NODES[src]["y"]),
        (NODES[dst]["x"], NODES[dst]["y"]),
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=MUTATION,
        color=color, linewidth=lw,
        shrinkA=0, shrinkB=0, zorder=1,
        patchA=circles[src], patchB=circles[dst],
    )
    ax.add_patch(arrow)


def _draw_stubs(ax, circles):
    """Continuity stubs (dashed) into Z_{t-1} and out of Z_t."""
    zprev = NODES["Zprev"]
    z     = NODES["Z"]
    stub_start_x = zprev["x"] - STUB_LEN - R
    stub_end_x   = zprev["x"] - R
    head_room = 0.18 * NODE_DIAM
    ax.plot([stub_start_x, stub_end_x - head_room], [zprev["y"], zprev["y"]],
            color=STROKE_COLOR, linewidth=STUB_LINEW, linestyle=STUB_DASH,
            zorder=1, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        (stub_end_x - head_room, zprev["y"]),
        (zprev["x"], zprev["y"]),
        arrowstyle="-|>", mutation_scale=MUTATION,
        color=STROKE_COLOR, linewidth=STUB_LINEW,
        shrinkA=0, shrinkB=0, zorder=1,
        patchB=circles["Zprev"],
    ))
    ax.plot([z["x"] + R, z["x"] + R + STUB_LEN], [z["y"], z["y"]],
            color=STROKE_COLOR, linewidth=STUB_LINEW, linestyle=STUB_DASH,
            zorder=1, solid_capstyle="butt")


def render_step(spec, out_stem):
    fig, ax = _make_axes()

    # Title above the diagram
    ax.set_title(spec["title"], fontsize=14, pad=10, loc="left", color=STROKE_COLOR)

    # Nodes (highlight new ones)
    circles = {}
    for name in spec["nodes"]:
        circles[name] = _draw_node(ax, name, highlight=name in spec["new_nodes"])

    # Edges (highlight new ones)
    for (src, dst) in spec["edges"]:
        rad = ALL_EDGES[(src, dst)]
        _draw_edge(ax, src, dst, rad, circles,
                   highlight=(src, dst) in spec["new_edges"])

    if spec["stubs"]:
        _draw_stubs(ax, circles)

    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight",
                    pad_inches=0.1, facecolor="white")
    plt.close(fig)


def main():
    docs_dir = Path(__file__).parent.parent   # docs/
    out_dir = docs_dir / "figures" / "dag_buildup"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, spec in enumerate(_step_specs(), start=1):
        out = out_dir / f"step{i}"
        render_step(spec, str(out))
        written.append(f"{out}.png")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
