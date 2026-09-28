"""Real-data panels embedded in the TikZ architecture figure (figures/fig_arch.tex).

Window 2019-07-14 08:12 (test window 1073), the window also shown for B1. Every curve is
measured data from forecast_vs_observed_15min.csv. Panels are written at their final printed
size (cm) in the framed-box style of the architecture figure.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
CM = 1 / 2.54

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#6b6b6b",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.major.pad": 1.2,
    "ytick.major.pad": 1.2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
})
C_OURS, C_SPM, BAND, C_HIST = "#1F5FBF", "#E8692F", "#A9C8EE", "#9C5B2E"

df = pd.read_csv(os.path.join(ROOT, "forecast_vs_observed_15min.csv"), parse_dates=["valid_time"])
win = df[df.window_id == 1073].sort_values("lead_min")
t0 = win.valid_time.iloc[0] - pd.Timedelta(minutes=2)
day = df[df.valid_time.dt.date == t0.date()].drop_duplicates("valid_time").set_index("valid_time").sort_index()
hist_t = [t0 - pd.Timedelta(minutes=m) for m in range(14, -1, -2)]
assert max(np.min(np.abs((day.index - t).total_seconds())) for t in hist_t) <= 60
hist = np.array([day.observed_kw.iloc[np.argmin(np.abs((day.index - t).total_seconds()))] for t in hist_t])
hmin, lead = np.arange(-14, 1, 2), np.arange(2, 17, 2)
obs, med = win.observed_kw.values, win.ddim_median_kw.values
p05, p95, spm = win.ddim_p05_kw.values, win.ddim_p95_kw.values, win.spm_kw.values


def framed(w_cm, h_cm, l, b, r, t):
    fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
    ax = fig.add_axes([l / w_cm, b / h_cm, 1 - (l + r) / w_cm, 1 - (b + t) / h_cm])
    for sp in ax.spines.values():
        sp.set_visible(True)
    ax.axvline(0, color="#8a8a8a", lw=0.5, ls=(0, (1, 1.5)), zorder=0)
    ax.set_xlim(-15.5, 17.5)
    ax.set_ylim(2.3, 7.3)
    return fig, ax


# input: PV history, clear-sky smart-persistence anchor and the noisy residual draw (no ticks)
rng = np.random.default_rng(3)
fig, ax = framed(3.04, 1.04, 0.02, 0.02, 0.02, 0.02)
ax.plot(hmin, hist, color=C_HIST, marker="o", ms=1.8, lw=0.8)
ax.plot(lead, spm, color=C_SPM, ls=(0, (2.2, 1.2)), lw=0.9)
ax.scatter(lead, np.clip(spm + 0.9 * rng.standard_normal(8), 2.8, 6.9), s=3, color="#7a7a7a", zorder=3, lw=0)
ax.set_xticks([])
ax.set_yticks([])
fig.savefig(os.path.join(FIG, "arch_b1_pv_in.pdf"), transparent=True)
plt.close(fig)

# output: calibrated 30-member forecast (axis on the right)
fig, ax = framed(3.50, 1.42, 0.02, 0.36, 0.46, 0.04)
ax.fill_between(lead, p05, p95, color=BAND, lw=0)
ax.plot(hmin, hist, color="black", marker="o", ms=1.5, lw=0.6)
ax.plot(lead, spm, color=C_SPM, ls=(0, (2.2, 1.2)), lw=0.8)
ax.plot(lead, med, color=C_OURS, lw=1.0)
ax.plot(lead, obs, ls="none", marker="o", ms=2.2, mfc="white", mec="black", mew=0.5)
ax.yaxis.tick_right()
ax.set_yticks([3, 5, 7])
ax.set_xticks([-14, 0, 16])
fig.savefig(os.path.join(FIG, "arch_b1_pv_out.pdf"), transparent=True)
plt.close(fig)
print("saved arch_b1_pv_in.pdf, arch_b1_pv_out.pdf | history", np.round(hist, 2))
