"""Builds the paper figures from the artefacts exported by ddim.ipynb (no model re-run)."""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Fixed categorical order: DDIM, SPM, Two-Stage; markers + dashes keep identity in greyscale print
C = {"ddim": "#2a78d6", "spm": "#eb6834", "two": "#1baf7a", "ref": "#52514e"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.4, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.5, "axes.edgecolor": "#52514e",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

d = pd.read_csv(os.path.join(ROOT, "forecast_vs_observed_15min.csv"), parse_dates=["valid_time"])
d["width"] = d.ddim_p95_kw - d.ddim_p05_kw
lead = np.sort(d.lead_min.unique())
g = d.groupby("lead_min")
mae = {k: (d[c] - d.observed_kw).abs().groupby(d.lead_min).mean().values
       for k, c in [("ddim", "ddim_median_kw"), ("spm", "spm_kw"), ("two", "two_stage_kw")]}
cov = g.inside_90pct.mean().values * 100
width = g.width.mean().values

# Sky-image fidelity per lead, as printed by ddim.ipynb (Cell 22)
ssim_gen = [0.787, 0.719, 0.650, 0.595, 0.542, 0.488, 0.447, 0.423]
ssim_pers = [0.826, 0.792, 0.772, 0.758, 0.745, 0.735, 0.727, 0.717]

# DDIM ensemble CRPS per lead (ddim.ipynb Cell 21, crps_lead["Joint DDIM"]); the 30-member ensemble
# is not exported, so these were read off showcase_15min_dashboard.png with the axis calibrated on
# the SPM markers (residual < 0.001 kW). Their mean, 1.222 kW, matches results_15min.json.
crps_ddim = [0.806, 1.025, 1.142, 1.217, 1.338, 1.347, 1.418, 1.487]

# ---- Fig: lead-time diagnostics (5 panels, double column, one shared legend on top)
fig, ax = plt.subplots(1, 5, figsize=(7.16, 2.1))
# Deterministic baselines: CRPS equals MAE
for k, y, st in [("ddim", crps_ddim, "o-"), ("spm", mae["spm"], "s--"), ("two", mae["two"], "^:")]:
    ax[0].plot(lead, y, st, color=C[k])
ax[0].set(title="(a) CRPS by lead time", xlabel="Lead time (min)", ylabel="CRPS (kW)", ylim=(0.5, 2.3))

ax[1].plot(lead, cov, "o-", color=C["ddim"])
ax[1].axhline(90, color=C["ref"], ls="--", lw=1)
ax[1].set(title="(b) 90% interval coverage", xlabel="Lead time (min)", ylabel="Coverage (%)",
          ylim=(40, 100))

ax[2].plot(lead, width, "o-", color=C["ddim"])
ax[2].set(title="(c) 90% interval width", xlabel="Lead time (min)", ylabel="Width (kW)",
          ylim=(0, 7))

ax[3].plot(lead, ssim_gen, "o-", color=C["ddim"])
ax[3].plot(lead, ssim_pers, "s--", color=C["ref"])
ax[3].set(title="(d) Sky-image SSIM", xlabel="Lead time (min)", ylabel="SSIM", ylim=(0.15, 0.9))

for k, st in [("ddim", "o-"), ("spm", "s--"), ("two", "^:")]:
    ax[4].plot(lead, mae[k], st, color=C[k])
ax[4].set(title="(e) MAE by lead time", xlabel="Lead time (min)", ylabel="MAE (kW)", ylim=(0.5, 2.3))
for a in ax:
    a.set_xticks([2, 6, 10, 14])

h = [plt.Line2D([], [], color=C["ddim"], ls="-", marker="o"),
     plt.Line2D([], [], color=C["spm"], ls="--", marker="s"),
     plt.Line2D([], [], color=C["two"], ls=":", marker="^"),
     plt.Line2D([], [], color=C["ref"], ls="--", marker="s"),
     plt.Line2D([], [], color=C["ref"], ls="--", lw=1)]
fig.legend(h, ["Joint DDIM", "Smart persistence", "Two-stage", "Last-frame image persistence",
               "Nominal 90% coverage"], loc="upper center", ncol=5, frameon=False, handlelength=2.4,
           columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT, "lead_diagnostics.pdf"))
plt.close(fig)

# ---- Fig: day-long +16 min forecasts on six test days spanning the variability range
# Variability = mean |dP| between consecutive observations; the six days sit at evenly spaced ranks,
# so the least and the most variable test day are always the first and last panel.
obs_all = d.drop_duplicates("valid_time").sort_values("valid_time")
obs_day = obs_all.valid_time.dt.date
day_var = obs_all.observed_kw.groupby(obs_day).diff().abs().groupby(obs_day).mean().sort_values()
days = day_var.index[np.linspace(0, len(day_var) - 1, 6).round().astype(int)]
# single column (3.5 in): 3 x 2 panels, two-line panel titles, legend in two rows on top
fig, ax = plt.subplots(3, 2, figsize=(3.5, 4.4), sharey=True)
for i, (a, day) in enumerate(zip(ax.flat, days)):
    dd = d[d.valid_time.dt.date == day]
    obs = dd.drop_duplicates("valid_time").sort_values("valid_time")
    f = dd[dd.lead_min == lead.max()].sort_values("valid_time")
    a.plot(obs.valid_time, obs.observed_kw, color="#0b0b0b", lw=0.7, label="Observed PV")
    a.fill_between(f.valid_time, f.ddim_p05_kw, f.ddim_p95_kw, color=C["ddim"], alpha=0.22,
                   lw=0, label="DDIM 90% interval")
    a.plot(f.valid_time, f.ddim_median_kw, "-", color=C["ddim"], lw=0.9, label="DDIM median")
    a.plot(f.valid_time, f.spm_kw, "--", color=C["spm"], lw=0.7, label="Smart persistence")
    a.plot(f.valid_time, f.two_stage_kw, ":", color=C["two"], lw=0.9, label="Two-stage")
    m = (f.ddim_median_kw - f.observed_kw).abs().mean()
    a.set_title(f"({'abcdef'[i]}) {day}\nMAE {m:.2f} kW, cov. {f.inside_90pct.mean():.0%}",
                loc="left", fontsize=6.5, linespacing=1.1, pad=2)
    a.set_ylim(0, 30)
    a.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a.tick_params(labelsize=6)
for a in ax[:, 0]:
    a.set_ylabel("PV (kW)", fontsize=7)
for a in ax[-1]:
    a.set_xlabel("Local time", fontsize=7)
h, lab = ax[0, 0].get_legend_handles_labels()
fig.legend(h, lab, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6,
           handlelength=1.8, columnspacing=1.0)
fig.tight_layout(h_pad=0.3, w_pad=0.5, rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "daily_timeseries.pdf"))
plt.close(fig)

# ---- Sky-image panel: reuse the notebook render, cropping its embedded suptitle
im = Image.open(os.path.join(ROOT, "showcase_15min_sky_images.png")).convert("RGB")
w, h = im.size
im = im.crop((0, int(0.035 * h), w, h))
im.thumbnail((2400, 2400))
im.save(os.path.join(OUT, "sky_images.png"), optimize=True)

# ---- Architecture-figure assets: real frames and PV of the median-variability showcase window
# (issued 2019-07-14 08:12, "Typical" row of showcase_15min_sky_images.png).
ARCH = os.path.join(OUT, "arch")
os.makedirs(ARCH, exist_ok=True)
sky = np.asarray(Image.open(os.path.join(ROOT, "showcase_15min_sky_images.png")).convert("RGB"))


def run_starts(hits):
    """First index of each run of consecutive line positions (each axes frame line is 3 px thick)."""
    idx = np.where(hits)[0]
    return [int(v) for i, v in enumerate(idx) if i == 0 or v != idx[i - 1] + 1]


# every tile is bounded by a light-grey axes frame; long runs of frame pixels locate the tile edges
grey = sky.astype(int)
is_frame = (grey.mean(2) > 150) & (grey.mean(2) < 235) & (np.ptp(grey, axis=2) < 12)
rows = run_starts(is_frame.sum(1) > 0.5 * sky.shape[1])
cols = run_starts(is_frame.sum(0) > 0.3 * sky.shape[0])
tile_rows = list(zip(rows[0::2], rows[1::2]))  # clear obs, clear gen, typical obs, typical gen, ...
tile_cols = list(zip(cols[0::2], cols[1::2]))  # t=0, +2, ..., +16 min


def tile(r, c):
    """Recover the native 64x64 frame from a nearest-upsampled tile, then upsample x4 for print."""
    (y0, y1), (x0, x1) = tile_rows[r], tile_cols[c]
    t = sky[y0 + 3:y1, x0 + 3:x1]
    ys = ((np.arange(64) + 0.5) * t.shape[0] / 64).astype(int)
    xs = ((np.arange(64) + 0.5) * t.shape[1] / 64).astype(int)
    return Image.fromarray(t[ys][:, xs]).resize((256, 256), Image.NEAREST)


tile(2, 0).save(os.path.join(ARCH, "hist_t0.png"))
for k, c in [("02", 1), ("08", 4), ("16", 8)]:
    tile(2, c).save(os.path.join(ARCH, f"obs_{k}.png"))
    tile(3, c).save(os.path.join(ARCH, f"gen_{k}.png"))
rng = np.random.default_rng(42)
for k in range(3):  # x_T ~ N(0, I) mapped back to pixels with the inverse of I/127.5 - 1
    eps = np.clip((rng.standard_normal((64, 64, 3)) + 1) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(eps).resize((256, 256), Image.NEAREST).save(os.path.join(ARCH, f"noise_{k}.png"))

issue = pd.Timestamp("2019-07-14 08:12:10")
win = d[(d.valid_time - pd.to_timedelta(d.lead_min, unit="min")) == issue].sort_values("lead_min")
hist = (d.drop_duplicates("valid_time").set_index("valid_time").observed_kw.sort_index()
        [issue - pd.Timedelta(minutes=14):issue])
assert len(win) == 8 and len(hist) == 8
hist_min = np.arange(-14, 1, 2)
pd.DataFrame({"min": hist_min, "kw": hist.values}).to_csv(os.path.join(ARCH, "pv_hist.csv"), index=False)
win.assign(min=win.lead_min).rename(columns={"observed_kw": "obs", "ddim_median_kw": "med"})[
    ["min", "obs", "med"]].to_csv(os.path.join(ARCH, "pv_fut.csv"), index=False)
band = pd.DataFrame({"min": np.r_[win.lead_min, win.lead_min[::-1]],
                     "kw": np.r_[win.ddim_p95_kw, win.ddim_p05_kw[::-1]]})
band.to_csv(os.path.join(ARCH, "pv_band.csv"), index=False)
# network input in its own scale: history p/(P_max/2) - 1, future replaced by x_T ~ N(0, 1)
pd.DataFrame({"min": hist_min, "v": hist.values / 15.0 - 1}).to_csv(os.path.join(ARCH, "pv_in_hist.csv"),
                                                                    index=False)
pd.DataFrame({"min": np.arange(2, 17, 2), "v": rng.standard_normal(8)}).to_csv(
    os.path.join(ARCH, "pv_in_noise.csv"), index=False)

print("MAE by lead:", {k: np.round(v, 3).tolist() for k, v in mae.items()})
print("coverage:", np.round(cov, 1).tolist(), "width:", np.round(width, 2).tolist())
