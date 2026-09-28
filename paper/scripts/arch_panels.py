"""Real-data panels embedded in the TikZ architecture figure (figures/fig_arch.tex).

Window 2019-07-14 08:12 (test window 1073). Every curve is measured data from
forecast_vs_observed_15min.csv; the uncalibrated band in arch2_calib.pdf is recovered exactly
from the calibrated 5/95 % quantiles by inverting eq. (6) with the fitted spread factors s_k
(no clipping is active for this window: all values lie far inside [0, 30] kW).
Panels are written at their final printed size (cm) so fonts stay at 6 pt in the paper.
"""
import json
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
    "font.size": 6,
    "axes.linewidth": 0.45,
    "xtick.major.width": 0.45,
    "ytick.major.width": 0.45,
    "xtick.major.size": 1.8,
    "ytick.major.size": 1.8,
    "xtick.major.pad": 1.2,
    "ytick.major.pad": 1.2,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})
C_OURS, C_SPM, BAND, BAND_RAW, C_PV = "#1F5FBF", "#E8692F", "#A9C8EE", "#3E7DD1", "#C8651B"

df = pd.read_csv(os.path.join(ROOT, "forecast_vs_observed_15min.csv"), parse_dates=["valid_time"])
res = json.load(open(os.path.join(ROOT, "results_15min.json")))
WID = 1073
win = df[df.window_id == WID].sort_values("lead_min")
t0 = win.valid_time.iloc[0] - pd.Timedelta(minutes=2)
day = df[df.valid_time.dt.date == t0.date()].drop_duplicates("valid_time").set_index("valid_time").sort_index()
hist_t = [t0 - pd.Timedelta(minutes=m) for m in range(14, -1, -2)]
gap = [np.min(np.abs((day.index - t).total_seconds())) for t in hist_t]
assert max(gap) <= 60
hist = np.array([day.observed_kw.iloc[np.argmin(np.abs((day.index - t).total_seconds()))] for t in hist_t])
hmin, lead = np.arange(-14, 1, 2), np.arange(2, 17, 2)
obs, med = win.observed_kw.values, win.ddim_median_kw.values
p05, p95, spm = win.ddim_p05_kw.values, win.ddim_p95_kw.values, win.spm_kw.values
s_k = np.array(res["calibration_scales"])
raw05, raw95 = med + (p05 - med) / s_k, med + (p95 - med) / s_k


def axes(w_cm, h_cm, l=0.62, b=0.42, r=0.08, t=0.08):
    fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
    ax = fig.add_axes([l / w_cm, b / h_cm, 1 - (l + r) / w_cm, 1 - (b + t) / h_cm])
    return fig, ax


def finish(fig, ax, name, ylim=(2, 8)):
    ax.axvline(0, color="#9a9a9a", lw=0.45, zorder=0)
    ax.set_xlim(-15, 17)
    ax.set_ylim(*ylim)
    ax.set_xticks([-14, 0, 16])
    ax.set_yticks([2, 5, 8])
    fig.savefig(os.path.join(FIG, name), transparent=True)
    plt.close(fig)
    print("saved", name)


# (a) input PV panel: history, SPM anchor over the horizon, noisy residual draw
rng = np.random.default_rng(3)
fig, ax = axes(3.30, 1.50, b=0.36)
ax.plot(hmin, hist, color="black", marker="o", ms=1.6, lw=0.7)
ax.plot(lead, spm, color=C_SPM, ls=(0, (2.2, 1.2)), lw=0.8)
ax.scatter(lead, np.clip(spm + 0.8 * rng.standard_normal(8), 4.0, 6.6), s=2.2, color="#8a8a8a", zorder=3, lw=0)
ax.set_ylabel("kW", labelpad=0.5)
ax.text(-14, 7.35, "history $P$", fontsize=5.8, va="center")
ax.text(16.5, 7.35, r"anchor $\hat P^{\rm SPM}$", color=C_SPM, fontsize=5.8, va="center", ha="right")
ax.text(16.5, 2.85, r"noisy $x^P_\tau$", color="#6f6f6f", fontsize=5.8, va="center", ha="right")
finish(fig, ax, "arch2_pv_in.pdf")

# (a) output panel: calibrated 30-member forecast
fig, ax = axes(4.30, 1.70, b=0.56)
ax.fill_between(lead, p05, p95, color=BAND, lw=0)
ax.plot(hmin, hist, color="black", marker="o", ms=1.5, lw=0.6)
ax.plot(lead, spm, color=C_SPM, ls=(0, (2.2, 1.2)), lw=0.7)
ax.plot(lead, med, color=C_OURS, lw=1.0)
ax.plot(lead, obs, ls="none", marker="o", ms=2.1, mfc="white", mec="black", mew=0.5)
ax.set_ylabel("kW", labelpad=0.5)
ax.set_xlabel("minutes from $t$", labelpad=0.5)
finish(fig, ax, "arch2_pv_out.pdf")

# (c) clear-sky residual target on the real window
fig, ax = axes(4.30, 2.05, b=0.56)
ax.plot(hmin, hist, color="black", marker="o", ms=1.6, lw=0.7, label="history $P$")
ax.plot(np.r_[0, lead], np.r_[hist[-1], spm], color=C_SPM, ls=(0, (2.2, 1.2)), lw=0.9,
        label=r"anchor $\hat P^{\rm SPM}$")
ax.plot(lead, obs, ls="none", marker="o", ms=2.3, mfc="white", mec="black", mew=0.55, label="future $P$")
for x, a_, o_ in zip(lead, spm, obs):
    ax.annotate("", xy=(x, o_ - 0.08), xytext=(x, a_ + 0.02),
                arrowprops=dict(arrowstyle="-|>,head_length=0.22,head_width=0.10", lw=0.7, color=C_PV, shrinkA=0, shrinkB=0))
ax.text(16.6, 2.85, r"residual $s_P\,x^P_0=P-\hat P^{\rm SPM}$", color=C_PV, fontsize=5.8, ha="right")
ax.legend(loc="upper left", frameon=False, fontsize=5.6, handlelength=1.6, borderaxespad=0.1,
          labelspacing=0.2, handletextpad=0.4)
ax.set_xlabel("minutes from $t$", labelpad=0.5)
ax.set_ylabel("kW", labelpad=0.5)
ax.axvline(0, color="#9a9a9a", lw=0.45, zorder=0)
ax.set_xlim(-15, 17)
ax.set_ylim(2.6, 7.4)
ax.set_xticks([-14, 0, 16])
ax.set_yticks([3, 5, 7])
fig.savefig(os.path.join(FIG, "arch2_resid.pdf"), transparent=True)
plt.close(fig)
print("saved arch2_resid.pdf")

# (d) spread calibration: raw vs calibrated 90 % band (same window, same members)
fig, ax = axes(2.95, 2.05, l=0.50, b=0.56)
ax.fill_between(lead, p05, p95, color=BAND, lw=0, label="calibrated")
ax.fill_between(lead, raw05, raw95, color=BAND_RAW, lw=0, alpha=0.9, label="raw")
ax.plot(lead, med, color=C_OURS, lw=0.9)
ax.plot(lead, obs, ls="none", marker="o", ms=2.1, mfc="white", mec="black", mew=0.5)
ax.set_xlim(1, 17)
ax.set_ylim(3.5, 7.5)
ax.set_xticks([2, 8, 16])
ax.set_yticks([4, 6])
ax.set_xlabel("lead (min)", labelpad=0.5)
ax.legend(loc="upper left", frameon=False, fontsize=5.6, handlelength=1.2, borderaxespad=0.1,
          labelspacing=0.2, handletextpad=0.4)
fig.savefig(os.path.join(FIG, "arch2_calib.pdf"), transparent=True)
plt.close(fig)
print("saved arch2_calib.pdf | s_k:", s_k.tolist())
print("history", np.round(hist, 2), "| raw 90% width", np.round(raw95 - raw05, 2), "| cal", np.round(p95 - p05, 2))
