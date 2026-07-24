"""Render a paper-quality architecture diagram of MTPNet (MTPNetRiskPredictor).

The figure is generated strictly from the source in models/mtpnet.py and shows
every PyTorch primitive module (nn.Linear / nn.LayerNorm / nn.GELU / nn.Dropout /
nn.Sigmoid / nn.GRU / softmax / element-wise ops) together with tensor shapes.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# --------------------------------------------------------------------------- #
# Color palette (clean, print-friendly)
# --------------------------------------------------------------------------- #
C_INPUT = "#D7E4F4"   # input / output tensors
C_LINEAR = "#A8C8EC"  # nn.Linear
C_NORM = "#C5E0C5"    # nn.LayerNorm
C_ACT = "#FFE3B0"     # nn.GELU / nn.Sigmoid / softmax
C_DROP = "#F6C6C6"    # nn.Dropout
C_RECUR = "#D2C2E8"   # nn.GRU
C_OP = "#FFF3A3"      # element-wise op (mul / add / sum)
C_GROUP_EDGE = "#9AA0A6"
C_TEXT = "#1A1A1A"
C_FLOW = "#3C4043"
C_SKIP = "#7E57C2"    # residual / gating accent

# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
BW = 2.7   # box width
BH = 0.62  # box height


def op_box(ax, x, y, label, sub=None, color=C_LINEAR, w=BW, h=BH):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="#5F6368",
        facecolor=color,
    )
    ax.add_patch(box)
    if sub:
        ax.text(x, y + 0.10, label, ha="center", va="center",
                fontsize=9.2, color=C_TEXT, fontweight="bold")
        ax.text(x, y - 0.14, sub, ha="center", va="center",
                fontsize=7.8, color="#3C4043")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9.2, color=C_TEXT, fontweight="bold")
    return (x, y)


def op_circle(ax, x, y, label, color=C_OP, r=0.30):
    circ = Circle((x, y), r, linewidth=1.0, edgecolor="#5F6368", facecolor=color)
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=12, color=C_TEXT, fontweight="bold")
    return (x, y)


def tensor_node(ax, x, y, label, sub=None):
    w, h = 3.6, 0.80
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4,
        edgecolor="#2A2A2A",
        facecolor=C_INPUT,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.13, label, ha="center", va="center",
            fontsize=10.5, color=C_TEXT, fontweight="bold")
    if sub:
        ax.text(x, y - 0.16, sub, ha="center", va="center",
                fontsize=8.4, color="#3C4043", style="italic")
    return (x, y)


def arrow(ax, p_from, p_to, rad=0.0, color=C_FLOW, lw=1.3, ls="-"):
    ar = FancyArrowPatch(
        p_from,
        p_to,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(ar)


def group(ax, x_lo, y_lo, x_hi, y_hi, title, color="#F1F3F4"):
    rect = FancyBboxPatch(
        (x_lo, y_lo),
        x_hi - x_lo,
        y_hi - y_lo,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        linewidth=1.3,
        edgecolor=C_GROUP_EDGE,
        facecolor=color,
        linestyle=(0, (6, 4)),
    )
    ax.add_patch(rect)
    ax.text(x_lo + 0.18, y_hi - 0.30, title, ha="left", va="center",
            fontsize=10, color="#202124", fontweight="bold", style="italic")


def shape_tag(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=7.8, color="#5F6368", style="italic",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#CCCCCC", lw=0.6))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 19))
ax.set_xlim(0, 10)
ax.set_ylim(-7.5, 21.5)
ax.axis("off")

ax.text(5, 21.0,
        "MTPNet — Mechanism Temporal Prediction Network",
        ha="center", va="center", fontsize=15, fontweight="bold", color=C_TEXT)
ax.text(5, 20.55,
        "MTPNetRiskPredictor  (input_size=F, hidden_size=H, "
        "num_layers=2, dropout=d)",
        ha="center", va="center", fontsize=9.5, color="#3C4043", style="italic")

CX = 5.0           # main column x
LX = 2.55          # left branch x
RX = 7.45          # right branch x
SHX = 9.05         # shape-tag x

# ---- Input ---------------------------------------------------------------- #
y_in = 19.5
tensor_node(ax, CX, y_in, "Input  x", "mechanism features  (B, T, F_in)")
arrow(ax, (CX, y_in - 0.40), (CX, 18.55))
shape_tag(ax, SHX, y_in, "(B, T, F_in)")

# ---- MechanismFeatureEncoder group ---------------------------------------- #
y_enc_top = 18.55
y_enc_bot = 14.20
group(ax, 0.9, y_enc_bot, 9.6, y_enc_top,
      "MechanismFeatureEncoder", color="#F3F6FB")

# Projection branch (left)
y_lp = 17.95
op_box(ax, LX, y_lp, "nn.Linear", "F_in → H", color=C_LINEAR)
y_ln1 = 17.25
op_box(ax, LX, y_ln1, "nn.LayerNorm", "H", color=C_NORM)
y_g1 = 16.55
op_box(ax, LX, y_g1, "nn.GELU", "()", color=C_ACT)
y_d1 = 15.85
op_box(ax, LX, y_d1, "nn.Dropout", "p = d", color=C_DROP)
arrow(ax, (CX, 18.40), (LX, y_lp + BH / 2))
arrow(ax, (LX, y_lp - BH / 2), (LX, y_ln1 + BH / 2))
arrow(ax, (LX, y_ln1 - BH / 2), (LX, y_g1 + BH / 2))
arrow(ax, (LX, y_g1 - BH / 2), (LX, y_d1 + BH / 2))

# Gate branch (right)
y_lg = 17.95
op_box(ax, RX, y_lg, "nn.Linear", "F_in → H", color=C_LINEAR)
y_sg = 17.10
op_box(ax, RX, y_sg, "nn.Sigmoid", "()", color=C_ACT)
arrow(ax, (CX, 18.40), (RX, y_lg + BH / 2))
arrow(ax, (RX, y_lg - BH / 2), (RX, y_sg + BH / 2))

# Element-wise multiply (GLU-style gating)
y_mul = 14.85
op_circle(ax, CX, y_mul, "⊙")
arrow(ax, (LX, y_d1 - BH / 2), (CX - 0.28, y_mul + 0.22), rad=0.12)
arrow(ax, (RX, y_sg - BH / 2), (CX + 0.28, y_mul + 0.22), rad=-0.12)
shape_tag(ax, SHX, y_mul, "(B, T, H)")
arrow(ax, (CX, y_mul - 0.30), (CX, 13.70))

# ---- Temporal Encoder: nn.GRU -------------------------------------------- #
y_gru_top = 13.70
y_gru_bot = 11.85
group(ax, 1.4, y_gru_bot, 8.6, y_gru_top,
      "Temporal Encoder", color="#F6F2FB")
y_gru = 12.78
op_box(ax, CX, y_gru, "nn.GRU",
       "input=H, hidden=H, layers=2, recurrent dropout=d",
       color=C_RECUR, w=5.4, h=0.95)
ax.text(CX, y_gru_bot + 0.18, "batch_first=True   →   h_t = GRU(x_t, h_{t-1})",
        ha="center", va="center", fontsize=7.6, color="#5F6368", style="italic")
shape_tag(ax, SHX, y_gru, "(B, T, H)")
arrow(ax, (CX, y_gru_bot), (CX, 11.20))

# ---- TemporalResidualBlock ------------------------------------------------ #
y_rb_top = 11.20
y_rb_bot = 5.95
group(ax, 1.4, y_rb_bot, 8.6, y_rb_top,
      "TemporalResidualBlock    x + Dropout( Mix( LayerNorm(x) ) )",
      color="#F1F7F1")

y_ln2 = 10.55
op_box(ax, CX, y_ln2, "nn.LayerNorm", "H", color=C_NORM)
y_l3 = 9.85
op_box(ax, CX, y_l3, "nn.Linear", "H → 2H", color=C_LINEAR)
y_g2 = 9.15
op_box(ax, CX, y_g2, "nn.GELU", "()", color=C_ACT)
y_d2 = 8.45
op_box(ax, CX, y_d2, "nn.Dropout", "p = d", color=C_DROP)
y_l4 = 7.75
op_box(ax, CX, y_l4, "nn.Linear", "2H → H", color=C_LINEAR)
y_d3 = 7.05
op_box(ax, CX, y_d3, "nn.Dropout", "p = d", color=C_DROP)

y_add = 6.40
op_circle(ax, CX, y_add, "⊕")

arrow(ax, (CX, y_rb_top), (CX, y_ln2 + BH / 2))
arrow(ax, (CX, y_ln2 - BH / 2), (CX, y_l3 + BH / 2))
arrow(ax, (CX, y_l3 - BH / 2), (CX, y_g2 + BH / 2))
arrow(ax, (CX, y_g2 - BH / 2), (CX, y_d2 + BH / 2))
arrow(ax, (CX, y_d2 - BH / 2), (CX, y_l4 + BH / 2))
arrow(ax, (CX, y_l4 - BH / 2), (CX, y_d3 + BH / 2))
arrow(ax, (CX, y_d3 - BH / 2), (CX, y_add + 0.30))

# residual skip connection (identity) routed on the left
arrow(ax, (CX - 0.6, y_rb_top), (0.85, y_rb_top), color=C_SKIP, lw=1.4)
arrow(ax, (0.85, y_rb_top), (0.85, y_add), color=C_SKIP, lw=1.4)
arrow(ax, (0.85, y_add), (CX - 0.30, y_add), color=C_SKIP, lw=1.4)
ax.text(0.55, (y_rb_top + y_add) / 2, "residual identity",
        ha="center", va="center", fontsize=7.8, color=C_SKIP, style="italic",
        rotation=90)

shape_tag(ax, SHX, y_add, "(B, T, H)")
arrow(ax, (CX, y_add - 0.30), (CX, 5.70))

# ---- Temporal Attention --------------------------------------------------- #
y_at_top = 5.70
y_at_bot = 2.80
group(ax, 1.4, y_at_bot, 8.6, y_at_top,
      "Temporal Attention Pooling", color="#FBF6EC")
y_ln3 = 5.10
op_box(ax, CX, y_ln3, "nn.LayerNorm", "H", color=C_NORM)
y_l5 = 4.40
op_box(ax, CX, y_l5, "nn.Linear", "H → 1", color=C_LINEAR)
y_sm = 3.70
op_box(ax, CX, y_sm, "softmax", "dim = 1  (over T)", color=C_ACT)
y_pool = y_at_bot + 0.30
op_circle(ax, CX, y_pool, "Σ")

arrow(ax, (CX, y_at_top), (CX, y_ln3 + BH / 2))
arrow(ax, (CX, y_ln3 - BH / 2), (CX, y_l5 + BH / 2))
arrow(ax, (CX, y_l5 - BH / 2), (CX, y_sm + BH / 2))
arrow(ax, (CX, y_sm - BH / 2), (CX, y_pool + 0.30))
# attention weights × value → weighted sum
arrow(ax, (CX + 0.30, y_pool), (8.6, y_pool), color=C_SKIP, lw=1.0)
ax.text(8.55, y_pool + 0.24, "Σ_t  a_t · h_t",
        ha="right", va="center", fontsize=7.6, color=C_SKIP, style="italic")

shape_tag(ax, SHX, y_pool, "(B, H)")
arrow(ax, (CX, y_pool - 0.30), (CX, 2.50))

# ---- Head ----------------------------------------------------------------- #
y_hd_top = 2.50
y_hd_bot = -2.20
group(ax, 1.4, y_hd_bot, 8.6, y_hd_top,
      "Risk Head   (nn.Sequential)", color="#F4F1F7")
y_ln4 = 1.90
op_box(ax, CX, y_ln4, "nn.LayerNorm", "H", color=C_NORM)
y_d4 = 1.20
op_box(ax, CX, y_d4, "nn.Dropout", "p = d", color=C_DROP)
y_l6 = 0.50
op_box(ax, CX, y_l6, "nn.Linear", "H → H/2", color=C_LINEAR)
y_g3 = -0.20
op_box(ax, CX, y_g3, "nn.GELU", "()", color=C_ACT)
y_d5 = -0.90
op_box(ax, CX, y_d5, "nn.Dropout", "p = d", color=C_DROP)
y_l7 = -1.60
op_box(ax, CX, y_l7, "nn.Linear", "H/2 → 1", color=C_LINEAR)

arrow(ax, (CX, y_hd_top), (CX, y_ln4 + BH / 2))
arrow(ax, (CX, y_ln4 - BH / 2), (CX, y_d4 + BH / 2))
arrow(ax, (CX, y_d4 - BH / 2), (CX, y_l6 + BH / 2))
arrow(ax, (CX, y_l6 - BH / 2), (CX, y_g3 + BH / 2))
arrow(ax, (CX, y_g3 - BH / 2), (CX, y_d5 + BH / 2))
arrow(ax, (CX, y_d5 - BH / 2), (CX, y_l7 + BH / 2))

# ---- output --------------------------------------------------------------- #
y_out_logit = -2.70
tensor_node(ax, CX, y_out_logit, "Logit  ŷ", "(B, 1)")
arrow(ax, (CX, y_l7 - BH / 2), (CX, y_out_logit + 0.40))
shape_tag(ax, SHX, y_out_logit, "(B, 1)")

ax.text(CX, y_out_logit - 0.80,
        "Training:  BCEWithLogitsLoss(ŷ, y)      "
        "Inference:  σ(ŷ) → risk probability",
        ha="center", va="center", fontsize=8.8, color="#3C4043", style="italic")

# ---- legend --------------------------------------------------------------- #
legend_items = [
    ("nn.Linear", C_LINEAR),
    ("nn.LayerNorm", C_NORM),
    ("Activation (GELU / Sigmoid / softmax)", C_ACT),
    ("nn.Dropout", C_DROP),
    ("nn.GRU", C_RECUR),
    ("element-wise op  (⊙ / ⊕ / Σ)", C_OP),
    ("tensor (input / output)", C_INPUT),
]
lx = 0.6
ly = -4.5
ax.text(lx, ly + 0.35, "Legend", fontsize=9.5, fontweight="bold", color=C_TEXT)
for i, (name, col) in enumerate(legend_items):
    yy = ly - 0.05 - i * 0.42
    ax.add_patch(Rectangle((lx, yy - 0.14), 0.38, 0.28, facecolor=col,
                           edgecolor="#5F6368", lw=0.8))
    ax.text(lx + 0.55, yy, name, fontsize=8.2, va="center", color=C_TEXT)

# hyperparameter note
ax.text(6.6, ly + 0.35, "Defaults",
        fontsize=9.5, fontweight="bold", color=C_TEXT, ha="left")
notes = [
    "F_in = 8  (mechanism features)",
    "H = 64  (hidden_size)",
    "num_layers = 2",
    "dropout d = 0.1",
    "head_hidden = H/2 = 32",
    "window T  (sliding window)",
    "pos_weight = N⁻ / N⁺",
]
for i, line in enumerate(notes):
    ax.text(6.6, ly - 0.05 - i * 0.42, line,
            fontsize=8.2, va="center", color="#3C4043", ha="left")

plt.tight_layout()
out_path = "/workspace/mtpnet_architecture.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
