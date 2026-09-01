"""Genere les figures du README  (a executer a la racine du projet)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "docs"
os.makedirs(OUT, exist_ok=True)

# ---------------- Fig 1 : architecture complete ----------------
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

pipe = [
    (1, 7, "Connectome\nbiologique", "#e8f4ff"),
    (1, 4.8, "EEG de\nreference", "#f0fff0"),
    (2.8, 7, "Hamiltonien\nfermionique", "#e8f4ff"),
    (2.8, 4.8, "Solveur\nquantique", "#fff2e0"),
    (4.6, 6, "Persistance\ntopologique H1/H2", "#f5f0ff"),
    (6.4, 6, "Tryperposition\ncollapse + p_n", "#ffe6e6"),
    (8.2, 5.5, "Signal\ncognitif calibre", "#e0ffe0"),
    (10, 5.5, "Validation\nPSD + LZ + ISO", "#e0ffe0"),
    (11.8, 5.5, "Hash-chain\nBLAKE3 -> SHA256", "#fff6d0"),
]
for x, y, txt, c in pipe:
    box = FancyBboxPatch(
        (x - 0.75, y - 0.65), 1.5, 1.3,
        boxstyle="round,pad=0.08", facecolor=c,
        edgecolor="#555", linewidth=1.4)
    ax.add_patch(box)
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.5,
            fontweight="bold")

seq = [(pipe[0], pipe[2]), (pipe[2], pipe[4]), (pipe[4], pipe[5]),
       (pipe[5], pipe[6]), (pipe[6], pipe[7]), (pipe[7], pipe[8])]
for a, b in seq:
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    arr = FancyArrowPatch(
        (x1 + 0.75, y1), (x2 - 0.75, y2),
        arrowstyle="->", mutation_scale=14, color="#555", linewidth=1.8)
    ax.add_patch(arr)

ax.text(6.5, 7.8, "PHASES 0-4 : PIPELINE QUANTIQUE", ha="center",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#888"))

p2 = [
    (3.5, 2.0, "SNN AdEx\nmicrocircuits\n80/20 E/I", "#ffe6e6"),
    (6, 2.0, "STDP triplet\n+ Betti H1\nmasque", "#f5f0ff"),
    (8.5, 2.0, "ISO-learning\nmeta-boucle\n(i_ext, depth)", "#e8f4ff"),
    (11, 2.0, "Readout\nreservoir\n(ridge taps)", "#e0ffe0"),
]
for x, y, txt, c in p2:
    box = FancyBboxPatch(
        (x - 1.1, y - 0.75), 2.2, 1.5,
        boxstyle="round,pad=0.08", facecolor=c,
        edgecolor="#555", linewidth=1.4)
    ax.add_patch(box)
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.5,
            fontweight="bold")

for i in range(3):
    x1, y1 = p2[i][0], p2[i][1]
    x2, y2 = p2[i + 1][0], p2[i + 1][1]
    arr = FancyArrowPatch(
        (x1 + 1.1, y1), (x2 - 1.1, y2),
        arrowstyle="->", mutation_scale=14, color="#555", linewidth=1.8)
    ax.add_patch(arr)

ax.text(7.5, 0.6, "PHASE 4b : DYNAMIQUE CAUSALE + APPRENTISSAGE",
        ha="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#888"))

plt.savefig(f"{OUT}/pipeline_architecture.png", dpi=110,
            bbox_inches="tight")
plt.close()
print("fig 1 ok")

# ---------------- Fig 2 : resultats ISO ----------------
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ["Surrogate\nstatistique", "SNN forward\n(sans entrainement)",
          "Readout reservoir\n(ridge, taps=2)"]
vals = [0.419, -0.144, 0.467]
colors = ["#6495ed", "#d94949", "#46b162"]
bars = ax.bar(labels, vals, color=colors, edgecolor="#333",
              linewidth=1.2, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03 if v > 0 else v - 0.08,
            f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("ISO (isomorphism micro-etats)", fontsize=11)
ax.set_title(
    "Validation honnete : ISO sur EEG reelle (determinisme x2 verifie)",
    fontsize=11, fontweight="bold")
ax.axhline(0, color="#333", linewidth=0.8)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-0.35, 0.6)
plt.tight_layout()
plt.savefig(f"{OUT}/iso_results.png", dpi=110, bbox_inches="tight")
plt.close()
print("fig 2 ok")

# ---------------- Fig 3 : boucle ISO-learning ----------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")
ax.add_patch(FancyBboxPatch((0.3, 3), 11.4, 3.2, boxstyle="round,pad=0.1",
                            facecolor="#f8f8f8", edgecolor="#666"))
ax.text(6, 6.2, "BOUCLE ISO-LEARNING (methode sans-gradient)",
        ha="center", fontweight="bold", fontsize=11)

nodes = [
    (2, 4.8, "echantillonne\n(i_ext, depth)", "#ffe6e6"),
    (4.5, 4.8, "simulate_snn\n(STDP triplet + bW)", "#f5f0ff"),
    (7, 4.8, "readout\nreservoir\nridge + taps", "#e0ffe0"),
    (9.5, 4.8, "evalue ISO\n(micro-etats)", "#e8f4ff"),
]
for x, y, txt, c in nodes:
    ax.add_patch(FancyBboxPatch((x - 1.0, y - 0.7), 2.0, 1.4,
                                boxstyle="round,pad=0.07", facecolor=c,
                                edgecolor="#555", linewidth=1.3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=9,
            fontweight="bold")

for i in range(3):
    x1, y1 = nodes[i][0], nodes[i][1]
    x2, y2 = nodes[i + 1][0], nodes[i + 1][1]
    arr = FancyArrowPatch((x1 + 1.0, y1), (x2 - 1.0, y2),
                          arrowstyle="->", mutation_scale=14,
                          color="#555", linewidth=1.6)
    ax.add_patch(arr)

arr = FancyArrowPatch((9.5, 4.0), (2, 4.0),
                      connectionstyle="arc3,rad=0.25",
                      arrowstyle="->", mutation_scale=14,
                      color="#c44", linewidth=1.6)
ax.add_patch(arr)
ax.text(5.7, 1.7, "mise a jour W : poids excitateur bornes [0..2]\n"
        "eligibility = w_final complet (pas de moyenne)",
        ha="center", fontsize=8.5, color="#c44")
ax.text(6, 0.8, "Epoch ISO avec drive = 0.059 (limite documentee) ; "
        "readout sur forward sans drive = 0.467",
        ha="center", fontsize=9, style="italic")

plt.tight_layout()
plt.savefig(f"{OUT}/iso_learning_loop.png", dpi=110, bbox_inches="tight")
plt.close()
print("fig 3 ok")
