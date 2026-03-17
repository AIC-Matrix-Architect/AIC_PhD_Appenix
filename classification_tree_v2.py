# =============================================================================
# FORMAL COMPLEX CLASSIFICATION TREE — v2 Corrected + 4-Test Resolved
# Produces a high-quality PNG diagram of the full classification tree.
#
# Dependencies: pip install matplotlib
# Output:       classification_tree_v2.png
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

FIG_W, FIG_H = 32, 26
DPI          = 180
OUT_FILE     = "classification_tree_v2.png"

C = {
    "bg":          "#0D1117",
    "title_box":   "#161B22",
    "border":      "#30363D",
    "white":       "#E6EDF3",
    "grey":        "#8B949E",
    "line":        "#58A6FF",
    "content":     "#21262D",
    "empty":       "#3D1F1F",
    "complex":     "#1F2D3D",
    "passive":     "#1A2E2A",
    "active":      "#1A2A3D",
    "adaptive":    "#2A1A3D",
    "null_pre":    "#3D2E1A",
    "null_post":   "#3D1A1A",
    "struct":      "#1A3D2E",
    "agent":       "#1A2A3D",
    "purpose":     "#2A2A1A",
    "aorg":        "#3D2A1A",
    "asye":        "#2A3D1A",
    "dis":         "#3D1A1A",
    "gate_4t":     "#2A1A3D",
    "sys":         "#1A3D1A",
    "asys_f":      "#3D3D1A",
    "b_content":   "#58A6FF",
    "b_empty":     "#FF6B6B",
    "b_complex":   "#58A6FF",
    "b_passive":   "#79C0FF",
    "b_active":    "#79C0FF",
    "b_adaptive":  "#D2A8FF",
    "b_null":      "#FFA657",
    "b_struct":    "#3FB950",
    "b_agent":     "#58A6FF",
    "b_purpose":   "#E3B341",
    "b_aorg":      "#FFA657",
    "b_asye":      "#3FB950",
    "b_dis":       "#FF6B6B",
    "b_gate":      "#D2A8FF",
    "b_sys":       "#3FB950",
    "b_asys_f":    "#E3B341",
}

# =============================================================================
# DRAWING HELPERS
# =============================================================================

def rbox(ax, cx, cy, w, h, text, fc, ec,
         fontsize=7.5, fontcolor="#E6EDF3", bold=False,
         radius=0.008, alpha=1.0, linestyle="-", lw=1.5):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=linestyle,
        zorder=3, alpha=alpha
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text,
            ha="center", va="center",
            fontsize=fontsize, color=fontcolor,
            fontweight=weight, zorder=4,
            fontfamily="monospace",
            wrap=False)


def arrow(ax, x0, y0, x1, y1,
          color="#58A6FF", lw=1.3, style="->",
          rad=0.0, shrink=3):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            shrinkA=shrink, shrinkB=shrink,
            connectionstyle=f"arc3,rad={rad}"
        ),
        zorder=2
    )


def vline(ax, x, y0, y1, color="#58A6FF", lw=1.2):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=2)


def hline(ax, x0, x1, y, color="#58A6FF", lw=1.2):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=2)


def connector(ax, px, py, cx, cy, color="#58A6FF", lw=1.2):
    mid_y = (py + cy) / 2
    vline(ax, px, py, mid_y, color=color, lw=lw)
    hline(ax, px, cx, mid_y, color=color, lw=lw)
    vline(ax, cx, mid_y, cy, color=color, lw=lw)


# =============================================================================
# MAIN DRAWING FUNCTION
# =============================================================================

def draw_tree():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    BW   = 0.055
    BH   = 0.038
    TBH  = 0.030
    WBW  = 0.070

    # TITLE
    rbox(ax, 0.50, 0.975, 0.82, 0.038,
         "FORMAL COMPLEX CLASSIFICATION TREE  —  v2 Corrected + 4-Test Resolved",
         C["title_box"], C["b_content"],
         fontsize=11, bold=True, radius=0.006)

    # LEGEND PANEL
    legend_y = 0.060
    legend_items = [
        ("AORG = AORGANISATIONAL",   C["aorg"],   C["b_aorg"]),
        ("ASYE = ASYSTEM EMBRYONIC", C["asye"],   C["b_asye"]),
        ("ASYF = ASYSTEM FUNCTIONAL",C["asys_f"], C["b_asys_f"]),
        ("SYS  = SYSTEM",            C["sys"],    C["b_sys"]),
        ("DIS  = DISORG",            C["dis"],    C["b_dis"]),
        ("[4T] = 4-CONSTRAINT GATE", C["gate_4t"],C["b_gate"]),
    ]
    legend_items2 = [
        ("ABST = ABSENT",            C["purpose"], C["b_purpose"]),
        ("EMRG = EMERGING",          C["purpose"], C["b_purpose"]),
        ("AMBG = AMBIGUOUS",         C["purpose"], C["b_purpose"]),
        ("CVAL = CLEAR_VALIDATED",   C["purpose"], C["b_purpose"]),
        ("STRUCT= STRUCTURAL OrgXP", C["struct"],  C["b_struct"]),
        ("AGENT = AGENTIVE OrgXP",   C["agent"],   C["b_agent"]),
    ]

    for i, (txt, fc, ec) in enumerate(legend_items):
        lx = 0.04 + i * 0.158
        rbox(ax, lx, legend_y, 0.148, 0.026, txt,
             fc, ec, fontsize=6.8, radius=0.004)

    for i, (txt, fc, ec) in enumerate(legend_items2):
        lx = 0.04 + i * 0.158
        rbox(ax, lx, legend_y - 0.035, 0.148, 0.026, txt,
             fc, ec, fontsize=6.8, radius=0.004)

    # LEVEL 0 — CONTENT
    N_content = (0.50, 0.918)
    rbox(ax, *N_content, WBW, BH, "CONTENT",
         C["content"], C["b_content"], fontsize=9, bold=True)

    # LEVEL 1 — EMPTY SET | COMPLEX
    N_empty   = (0.18, 0.855)
    N_complex = (0.72, 0.855)

    rbox(ax, *N_empty, BW*1.4, BH,
         "EMPTY SET\n(not a complex)",
         C["empty"], C["b_empty"], fontsize=7.5)

    rbox(ax, *N_complex, BW*1.6, BH,
         "COMPLEX\n(non-empty substance)",
         C["complex"], C["b_complex"], fontsize=7.5, bold=True)

    connector(ax, N_content[0], N_content[1]-BH/2,
              N_empty[0],   N_empty[1]+BH/2,   C["b_empty"])
    connector(ax, N_content[0], N_content[1]-BH/2,
              N_complex[0], N_complex[1]+BH/2, C["b_complex"])

    # LEVEL 2 — PASSIVE | ACTIVE | ADAPTIVE
    N_pass = (0.20, 0.785)
    N_act  = (0.52, 0.785)
    N_adap = (0.82, 0.785)

    rbox(ax, *N_pass, 0.10, BH,
         "PASSIVE\n(no agents;\nstructural rules)",
         C["passive"], C["b_passive"], fontsize=7)

    rbox(ax, *N_act, 0.11, BH,
         "ACTIVE\n(agents apply\nfixed rules)",
         C["active"], C["b_active"], fontsize=7)

    rbox(ax, *N_adap, 0.11, BH,
         "ADAPTIVE\n(agents modify\nrule application)",
         C["adaptive"], C["b_adaptive"], fontsize=7)

    branch_y_top = N_complex[1] - BH/2
    for nx_ in [N_pass[0], N_act[0], N_adap[0]]:
        connector(ax, N_complex[0], branch_y_top,
                  nx_, N_pass[1]+BH/2, C["line"])

    # LEVEL 3 — OrgXP MODE NODES
    orgxp_y = 0.700

    P_null  = (0.110, orgxp_y)
    P_str   = (0.245, orgxp_y)
    A_null  = (0.420, orgxp_y)
    A_str   = (0.520, orgxp_y)
    A_agt   = (0.620, orgxp_y)
    D_null  = (0.760, orgxp_y)
    D_agt   = (0.880, orgxp_y)

    def orgxp_node(ax, pos, kind, parent_x, parent_y):
        lbl_map = {
            "null": "NULL\nOrgXP",
            "str":  "STRUCT\nOrgXP",
            "agt":  "AGENT\nOrgXP",
        }
        fc_map  = {"null": C["null_pre"], "str": C["struct"], "agt": C["agent"]}
        ec_map  = {"null": C["b_null"],   "str": C["b_struct"],"agt": C["b_agent"]}
        rbox(ax, pos[0], pos[1], 0.075, BH,
             lbl_map[kind], fc_map[kind], ec_map[kind], fontsize=6.8)
        connector(ax, parent_x, parent_y - BH/2,
                  pos[0], pos[1]+BH/2, ec_map[kind])

    orgxp_node(ax, P_null, "null", N_pass[0], N_pass[1])
    orgxp_node(ax, P_str,  "str",  N_pass[0], N_pass[1])
    orgxp_node(ax, A_null, "null", N_act[0],  N_act[1])
    orgxp_node(ax, A_str,  "str",  N_act[0],  N_act[1])
    orgxp_node(ax, A_agt,  "agt",  N_act[0],  N_act[1])
    orgxp_node(ax, D_null, "null", N_adap[0], N_adap[1])
    orgxp_node(ax, D_agt,  "agt",  N_adap[0], N_adap[1])

    null_nodes = [P_null, A_null, D_null]
    for nn in null_nodes:
        ax.annotate("pre->AORG\npost->DIS",
                    xy=(nn[0], nn[1] - BH/2 - 0.012),
                    ha="center", va="top", fontsize=5.8,
                    color=C["b_null"],
                    fontfamily="monospace", zorder=4)

    # LEVEL 4 — PURPOSE COLUMNS
    purp_y = 0.580
    purp_lbls = ["ABST", "EMRG", "AMBG", "CVAL"]

    def purpose_row(ax, parent_x, parent_y, x_centre, spacing=0.030):
        xs = [x_centre + (i - 1.5) * spacing for i in range(4)]
        fc_map  = {"ABST": C["purpose"], "EMRG": C["purpose"],
                   "AMBG": C["purpose"], "CVAL": C["gate_4t"]}
        ec_map  = {"ABST": C["b_purpose"],"EMRG": C["b_purpose"],
                   "AMBG": C["b_purpose"], "CVAL": C["b_gate"]}
        for i, (lbl, x) in enumerate(zip(purp_lbls, xs)):
            rbox(ax, x, purp_y, 0.026, 0.028, lbl,
                 fc_map[lbl], ec_map[lbl], fontsize=6.0)
            connector(ax, parent_x, parent_y-BH/2,
                      x, purp_y+0.014, C["b_purpose"])
        return xs

    xs_pstr  = purpose_row(ax, P_str[0],  P_str[1],  P_str[0],  0.028)
    xs_astr  = purpose_row(ax, A_str[0],  A_str[1],  A_str[0],  0.028)
    xs_aagt  = purpose_row(ax, A_agt[0],  A_agt[1],  A_agt[0],  0.028)
    xs_dagt  = purpose_row(ax, D_agt[0],  D_agt[1],  D_agt[0],  0.028)

    # LEVEL 5 — VERDICT LEAF NODES
    verd_y = 0.495
    verd_map = {
        "ABST": ("AORG", C["aorg"],   C["b_aorg"]),
        "EMRG": ("ASYE", C["asye"],   C["b_asye"]),
        "AMBG": ("DIS",  C["dis"],    C["b_dis"]),
        "CVAL": ("[4T]", C["gate_4t"],C["b_gate"]),
    }

    def verdict_row(ax, xs):
        for lbl, x in zip(purp_lbls, xs):
            vtxt, vfc, vec = verd_map[lbl]
            rbox(ax, x, verd_y, 0.026, 0.026, vtxt,
                 vfc, vec, fontsize=6.0)
            vline(ax, x, purp_y-0.014, verd_y+0.013, vec)
        return xs

    verdict_row(ax, xs_pstr)
    verdict_row(ax, xs_astr)
    verdict_row(ax, xs_aagt)
    verdict_row(ax, xs_dagt)

    # LEVEL 6 — 4-CONSTRAINT GATE + FINAL VERDICTS
    gate_y  = 0.415
    final_y = 0.340

    def constraint_block(ax, cval_x):
        rbox(ax, cval_x, gate_y, 0.060, 0.030,
             "C1^C2^C3^C4?",
             C["gate_4t"], C["b_gate"], fontsize=6.0, bold=True)
        vline(ax, cval_x, verd_y-0.013, gate_y+0.015, C["b_gate"])

        sys_x = cval_x - 0.022
        rbox(ax, sys_x, final_y, 0.038, 0.028,
             "YES->SYS", C["sys"], C["b_sys"], fontsize=6.0, bold=True)

        asf_x = cval_x + 0.022
        rbox(ax, asf_x, final_y, 0.038, 0.028,
             "NO->ASYF", C["asys_f"], C["b_asys_f"], fontsize=6.0, bold=True)

        connector(ax, cval_x, gate_y-0.015,
                  sys_x, final_y+0.014, C["b_sys"])
        connector(ax, cval_x, gate_y-0.015,
                  asf_x, final_y+0.014, C["b_asys_f"])

    constraint_block(ax, xs_pstr[3])
    constraint_block(ax, xs_astr[3])
    constraint_block(ax, xs_aagt[3])
    constraint_block(ax, xs_dagt[3])

    # LEVEL LABELS (left margin)
    levels = [
        (0.918, "L0: Content"),
        (0.855, "L1: Is it a complex?"),
        (0.785, "L2: ComplexType"),
        (0.700, "L3: OrgXP Mode"),
        (0.580, "L4: Purpose Clarity"),
        (0.495, "L5: Verdict (purpose)"),
        (0.415, "L6: 4-Constraint Gate"),
        (0.340, "L7: Final Verdict"),
    ]
    for ly, ltxt in levels:
        ax.text(0.005, ly, ltxt,
                ha="left", va="center",
                fontsize=6.0, color=C["grey"],
                fontfamily="monospace", zorder=4,
                style="italic")
        hline(ax, 0.000, 0.007, ly, color=C["border"], lw=0.8)

    # ABBREVIATIONS FOOTER
    abbrev = (
        "Abbreviations:  AORG=AORGANISATIONAL  |  ASYE=ASYSTEM_EMBRYONIC  |  "
        "ASYF=ASYSTEM_FUNCTIONAL  |  SYS=SYSTEM  |  DIS=DISORG  |  "
        "STRUCT=STRUCTURAL  |  AGENT=AGENTIVE  |  [4T]=4-CONSTRAINT GATE  |  "
        "ABST=ABSENT  |  EMRG=EMERGING  |  AMBG=AMBIGUOUS  |  CVAL=CLEAR_VALIDATED"
    )
    ax.text(0.50, 0.015, abbrev,
            ha="center", va="center",
            fontsize=6.2, color=C["grey"],
            fontfamily="monospace", zorder=4)

    # HORIZONTAL DIVIDERS
    div_ys = [0.935, 0.830, 0.758, 0.668, 0.555, 0.470, 0.390]
    for dy in div_ys:
        hline(ax, 0.008, 0.992, dy, color=C["border"], lw=0.6)

    # SAVE
    plt.tight_layout(pad=0)
    plt.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {os.path.abspath(OUT_FILE)}")


if __name__ == "__main__":
    draw_tree()
