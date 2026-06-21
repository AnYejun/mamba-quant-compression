"""
Generate all three paper figures from collected experiment data.
Run locally: python3 gen_figures.py
"""
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import os

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

C_BLUE  = '#2a78b5'
C_TEAL  = '#4aab9e'
C_RED   = '#d64242'
C_GRAY  = '#888888'
C_LIGHT = '#aacce8'
C_MID   = '#5596cc'

# -----------------------------------------------------------------------
# Figure 1 — Dose-response curve
# Data: cell 8 outputs (N=32, all-layer og sweep, n=30)
# -----------------------------------------------------------------------
og_vals = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
recall  = [0.00, 0.30, 0.60, 0.73, 0.60, 0.53]
base    = 0.53

fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.plot(og_vals, recall, 'o-', color=C_BLUE, lw=2, ms=7, zorder=3, clip_on=False)
ax.axhline(base, ls='--', color=C_GRAY, lw=1.2, label=f'base (no gain) = {base}')
ax.axvline(0.8, ls=':', color=C_RED, lw=1.2, alpha=0.8)
ax.annotate('peak 0.73', xy=(0.8, 0.73), xytext=(0.74, 0.76),
            fontsize=8.5, color=C_RED,
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1))
ax.set_xlabel('downstream gain $g$')
ax.set_ylabel('target recall ($N{=}32$, $n{=}30$)')
ax.set_xlim(0.35, 1.05)
ax.set_ylim(-0.05, 0.88)
ax.set_xticks([0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(['0.4', '0.6', '0.7', '0.8', '0.9', '1.0'])
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(PAPER_DIR, 'nfig_dose.pdf'), bbox_inches='tight')
plt.close()
print('nfig_dose.pdf ✓')

# -----------------------------------------------------------------------
# Figure 2 — Layer-band localization across N=16/32/64
# Data: cell 14 (Exp A-1) — 8-layer band sweep
# -----------------------------------------------------------------------
bands = ['L0–7', 'L8–15', 'L16–23', 'L24–31', 'L32–39', 'L40–47']
# Δ recall vs base for each band at each N
deltas = {
    16: [-0.03, -0.07,  0.07,  0.03,  0.03,  0.00],
    32: [ 0.00,  0.00,  0.13,  0.07,  0.00,  0.00],
    64: [ 0.00, -0.20,  0.20,  0.05, -0.25,  0.00],
}
colors_N = {16: C_LIGHT, 32: C_MID, 64: C_BLUE}
labels_N = {16: '$N{=}16$', 32: '$N{=}32$', 64: '$N{=}64$'}
x = np.arange(len(bands))
w = 0.26

fig, ax = plt.subplots(figsize=(6.2, 3.3))
for i, (N, dvals) in enumerate(deltas.items()):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, dvals, w,
                  color=colors_N[N], label=labels_N[N],
                  edgecolor='white', lw=0.4, zorder=2)
    # bold outline on L16-23 bar
    ax.bar(x[2] + offset, dvals[2], w,
           color=colors_N[N], edgecolor='black', lw=1.3, zorder=3)

ax.axhline(0, color='black', lw=0.8, zorder=1)
# annotate the dominant bar
ax.annotate('+0.20', xy=(x[2] + w, 0.20), xytext=(x[2] + w + 0.35, 0.23),
            fontsize=8, color=C_BLUE,
            arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=0.8))

ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.set_ylabel('$\Delta$ recall vs.\ base')
ax.set_xlabel('8-layer band')
ax.set_ylim(-0.35, 0.32)
ax.legend(title='capacity $N$', loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(PAPER_DIR, 'nfig_layers.pdf'), bbox_inches='tight')
plt.close()
print('nfig_layers.pdf ✓')

# -----------------------------------------------------------------------
# Figure 3 — Write-quantization (boundary section)
# Data: cell 13 + tab:quant (N=16, n=30)
# target:     {16:0.53, 8:0.53, 4:0.03, 2:0.00, 1:0.00}
# distractor: {16:0.53, 8:0.50, 4:0.00, 2:0.00, 1:0.00}  (8-bit from tab:quant)
# -----------------------------------------------------------------------
bits            = [1, 2, 4, 8, 16]
target_rec      = [0.00, 0.00, 0.03, 0.53, 0.53]
distractor_rec  = [0.00, 0.00, 0.00, 0.50, 0.53]

fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.plot(bits, target_rec,     'o-', color=C_BLUE, lw=2, ms=7,
        label='target quantized', zorder=3)
ax.plot(bits, distractor_rec, 's--', color=C_RED,  lw=2, ms=7,
        label='distractors quantized', zorder=3)
ax.axhline(0.53, ls=':', color=C_GRAY, lw=1.2, label='base recall = 0.53')

# annotate the cliff
ax.annotate('', xy=(4, 0.03), xytext=(8, 0.53),
            arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2))
ax.text(5.5, 0.30, 'cliff\n(8→4 bit)', fontsize=8, color=C_BLUE, ha='center')

ax.set_xscale('log', base=2)
ax.set_xticks(bits)
ax.set_xticklabels([f'{b}-bit' for b in bits])
ax.set_xlabel('quantization precision (log$_2$ scale)')
ax.set_ylabel('target recall ($N{=}16$, $n{=}30$)')
ax.set_ylim(-0.05, 0.70)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.25, lw=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(PAPER_DIR, 'nfig_quant.pdf'), bbox_inches='tight')
plt.close()
print('nfig_quant.pdf ✓')

print('\nAll figures written to', PAPER_DIR)
