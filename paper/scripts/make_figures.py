"""Builds every data figure of the paper from the measured result files.

Inputs (repository root):
  forecast_vs_observed_15min.csv  per-window, per-lead forecasts of CSR-MMDiff (30 members, calibrated)
  results_15min.json              aggregate metrics, validation sweeps
  figures/tiles/*.png             sky-image tiles (scripts/extract_tiles.py)
  scripts/reference_curves.json   per-lead curves of the baselines (see that file for provenance)

Outputs (paper/figures): fig_lead.pdf, fig_days.pdf, fig_images.pdf, fig_sweep.pdf and the
small panels used inside the TikZ architecture figure (arch_*.pdf / arch_*.png).
"""
import json
import os

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
TILES = os.path.join(FIG, "tiles")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.25,
    "lines.markersize": 3.4,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})

C_OURS = "#1f5fbf"
C_B1 = "#8e44ad"
C_SPM = "#e8692f"
C_PERS = "#4d4d4d"
C_OBS = "black"
BAND = "#9fc2ee"

df = pd.read_csv(os.path.join(ROOT, "forecast_vs_observed_15min.csv"), parse_dates=["valid_time"])
df["date"] = df.valid_time.dt.date.astype(str)
res = json.load(open(os.path.join(ROOT, "results_15min.json")))
ref = json.load(open(os.path.join(HERE, "reference_curves.json")))
lead = np.arange(2, 17, 2)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("saved", name)


# =====================================================================
# Fig. lead: performance by lead time (same five panels as B1 Fig. 2)
# =====================================================================
g = df.groupby("lead_min")
cov = g.inside_90pct.mean().values * 100
width = (df.ddim_p95_kw - df.ddim_p05_kw).groupby(df.lead_min).mean().values
mae_ours = (df.ddim_median_kw - df.observed_kw).abs().groupby(df.lead_min).mean().values
mae_spm = (df.spm_kw - df.observed_kw).abs().groupby(df.lead_min).mean().values
crps_ours = np.array(ref["ours"]["crps_by_lead"])
b1 = ref["b1"]

fig, axs = plt.subplots(1, 5, figsize=(7.16, 1.42))
kw_o = dict(color=C_OURS, marker="o", ls="-", label="CSR-MMDiff (ours)")
kw_b = dict(color=C_B1, marker="D", ls="-.", ms=2.8, label="Joint DDIM v1 [B1]")
kw_s = dict(color=C_SPM, marker="s", ls="--", ms=3.0, label="Smart persistence")

ax = axs[0]
ax.plot(lead, crps_ours, **kw_o)
ax.plot(lead, b1["crps"], **kw_b)
ax.plot(lead, mae_spm, **kw_s)
ax.set(title="(a) CRPS", ylabel="CRPS (kW)", ylim=(0.4, 2.0))

ax = axs[1]
ax.axhline(90, color="k", ls=(0, (3, 2)), lw=0.8, label="Nominal 90%")
ax.plot(lead, cov, **kw_o)
ax.plot(lead, b1["coverage"], **kw_b)
ax.set(title="(b) 90% interval coverage", ylabel="Coverage (%)", ylim=(60, 100))

ax = axs[2]
ax.plot(lead, width, **kw_o)
ax.plot(lead, b1["width"], **kw_b)
ax.set(title="(c) 90% interval width", ylabel="Width (kW)", ylim=(0, 7))

ax = axs[3]
ax.plot(lead, ref["ours"]["ssim_gen"], **kw_o)
ax.plot(lead, b1["ssim"], **kw_b)
ax.plot(lead, ref["ours"]["ssim_persistence"], color=C_PERS, marker="s", ls=":", ms=2.8,
        label="Last-frame persistence")
ax.set(title="(d) Sky-image SSIM", ylabel="SSIM", ylim=(0.35, 0.9))

ax = axs[4]
ax.plot(lead, mae_ours, **kw_o)
ax.plot(lead, b1["mae_median"], **kw_b)
ax.plot(lead, mae_spm, **kw_s)
ax.set(title="(e) MAE of median", ylabel="MAE (kW)", ylim=(0.4, 2.2))

for ax in axs:
    ax.set_xticks([2, 6, 10, 14])
    ax.set_xlabel("Lead time (min)")
    ax.yaxis.labelpad = 1.5
handles, labels = [], []
for ax in axs:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.2),
           handlelength=2.6, columnspacing=1.6)
fig.subplots_adjust(wspace=0.5)
save(fig, "fig_lead.pdf")

# =====================================================================
# Fig. days: +16 min forecasts on the six days used in B1 Fig. 3
# =====================================================================
days = [("2018-06-23", 0.74, 93), ("2019-10-14", 1.02, 92), ("2018-02-16", 1.38, 71),
        ("2017-10-11", 2.21, 77), ("2019-10-19", 2.94, 62), ("2017-09-20", 3.12, 82)]
fig, axs = plt.subplots(3, 2, figsize=(3.5, 2.95), sharey=True)
day_stats = {}
for ax, (d, b1_mae, b1_cov), tag in zip(axs.ravel(), days, "abcdef"):
    dd = df[df.date == d]
    obs = dd.drop_duplicates("valid_time").sort_values("valid_time")
    k = dd[dd.lead_min == 16].sort_values("valid_time")
    ax.fill_between(k.valid_time, k.ddim_p05_kw, k.ddim_p95_kw, color=BAND, alpha=0.75, lw=0,
                    label="Ours 90% interval")
    ax.plot(obs.valid_time, obs.observed_kw, color=C_OBS, lw=0.75, label="Observed PV")
    ax.plot(k.valid_time, k.spm_kw, color=C_SPM, lw=0.75, ls="--", label="Smart persistence")
    ax.plot(k.valid_time, k.ddim_median_kw, color=C_OURS, lw=0.9, label="Ours median")
    mae = float((k.ddim_median_kw - k.observed_kw).abs().mean())
    cv = float(k.inside_90pct.mean() * 100)
    day_stats[d] = {"mae": mae, "cov": cv, "b1_mae": b1_mae, "b1_cov": b1_cov}
    ax.set_title(f"({tag}) {d}\nMAE {mae:.2f} [{b1_mae:.2f}] kW, cov. {cv:.0f} [{b1_cov}]%",
                 loc="left", fontsize=6.1, pad=2, linespacing=1.05)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[8, 12, 16]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(labelsize=6.2)
for ax in axs[:, 0]:
    ax.set_ylabel("PV (kW)")
for ax in axs[-1]:
    ax.set_xlabel("Local time")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.035),
           handlelength=1.8, columnspacing=1.0, fontsize=6.3)
fig.subplots_adjust(hspace=0.78, wspace=0.08)
save(fig, "fig_days.pdf")
json.dump(day_stats, open(os.path.join(HERE, "day_stats.json"), "w"), indent=1)


# =====================================================================
# Fig. images: observed vs generated sky images on identical windows
# =====================================================================
def tile(name):
    return np.asarray(Image.open(os.path.join(TILES, name)).convert("RGB"))


windows = [("clear", "(a) Clear sky (lowest variability)"),
           ("typical", "(b) Typical (median variability)"),
           ("cloudy", "(c) Cloudy (highest variability)")]
cols = [0, 2, 8, 16]
rows = [("obs", "Observed"), ("ours", "CSR-MMDiff"), ("b1", "v1 [B1]")]
tw, gap_b, gap_t, lab_w = 0.44, 0.10, 0.018, 0.30
b2w = 2 * tw + gap_t
W = lab_w + 3 * (4 * tw + 3 * gap_t) + 3 * gap_b + b2w
H = 3 * tw + 2 * gap_t + 0.30
fig = plt.figure(figsize=(W, H))


def add(x, y, img, title=None):
    ax = fig.add_axes([x / W, y / H, tw / W, tw / H])
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.3)
        s.set_color("#999999")
    if title:
        ax.set_title(title, fontsize=6.2, pad=1.5)
    return ax


x0 = lab_w
for w, wtitle in windows:
    for ci, c in enumerate(cols):
        x = x0 + ci * (tw + gap_t)
        for ri, (src, _) in enumerate(rows):
            y = H - 0.26 - (ri + 1) * tw - ri * gap_t
            if c == 0:
                if src == "obs":
                    add(x, y, tile(f"ours_{w}_obs_0.png"), "$t$ (last input)")
                continue
            name = f"ours_{w}_obs_{c // 2}.png" if src == "obs" else f"{src}_{w}_gen_{c // 2}.png"
            add(x, y, tile(name), f"+{c} min" if ri == 0 else None)
    fig.text((x0 + 2 * tw + 1.5 * gap_t) / W, (H - 0.05) / H, wtitle, ha="center", va="top", fontsize=6.8)
    x0 += 4 * tw + 3 * gap_t + gap_b
for ri, (_, rl) in enumerate(rows):
    y = H - 0.26 - (ri + 1) * tw - ri * gap_t + tw / 2
    fig.text((lab_w - 0.03) / W, y / H, rl, ha="right", va="center", fontsize=6.6, rotation=0)
# B2 inset: generated frames of our earlier model B2 (one member, a different window)
fig.text((x0 + b2w / 2) / W, (H - 0.05) / H, "(d) B2, one member", ha="center", va="top", fontsize=6.8)
ytop = H - 0.26 - tw
ymid = ytop - (tw + 2 * gap_t) * 0.5
for k, lbl in enumerate(["+3", "+7", "+11", "+15"]):
    r, c = divmod(k, 2)
    a = add(x0 + c * (tw + gap_t), ymid - r * (tw + gap_t) + 0.5 * tw, tile(f"b2_gen_{k}.png"))
    a.text(0.04, 0.96, lbl, transform=a.transAxes, color="white", fontsize=5.6, va="top", ha="left")
fig.text((x0 + b2w / 2) / W, (H - 0.26 - 3 * tw - 2 * gap_t + 0.01) / H, "different window",
         ha="center", va="bottom", fontsize=5.8, style="italic", color="#444444")
save(fig, "fig_images.pdf")

# =====================================================================
# Fig. sweep: validation tuning of the sampler (steps, eta, guidance) and spread calibration
# =====================================================================
fig, axs = plt.subplots(1, 3, figsize=(3.5, 1.28), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
st = pd.DataFrame(res["steps_sweep"])
ax = axs[0]
ax.plot(st.steps, st.crps, color=C_OURS, marker="o")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50])
ax.set_xticklabels(["5", "10", "20", "50"])
ax.minorticks_off()
ax.set(xlabel="DDIM steps $S$", ylabel="Val. CRPS (kW)", title="(a) Steps (ms/member)")
for x_, y_, ms_ in zip(st.steps, st.crps, st.ms_per_member):
    ax.annotate(f"{ms_:.0f} ms", (x_, y_), textcoords="offset points", xytext=(0, 3.5),
                ha="center", fontsize=5.6, color="#333333")
ax.set_ylim(0.605, 0.665)
ax.set_xlim(4, 64)
ax.axvline(20, color="#888888", lw=0.6, ls=":")

tu = pd.DataFrame(res["tuning"])
ax = axs[1]
labels = [r"$\eta$=0", r"$\eta$=.5", r"$\eta$=1", "$g$=.5", "$g$=1"]
colors = [C_OURS if i == 0 else "#a9bfe0" for i in range(len(tu))]
ax.bar(range(len(tu)), tu.crps, color=colors, width=0.65, edgecolor="none")
ax.set_ylim(0.58, 0.68)
ax.set_xticks(range(len(tu)))
ax.set_xticklabels(labels, rotation=45, fontsize=6.0)
ax.set(title=r"(b) $\eta$ and guidance $g$")
ax.grid(axis="x", visible=False)

ax = axs[2]
sc = res["calibration_scales"]
ax.bar(lead, sc, width=1.3, color=C_OURS, edgecolor="none")
ax.axhline(1.0, color="k", lw=0.6, ls=(0, (3, 2)))
ax.set(xlabel="Lead (min)", title="(c) Spread factor $s_k$", ylim=(0, 2.6))
ax.set_xticks([2, 6, 10, 14])
ax.grid(axis="x", visible=False)
fig.subplots_adjust(wspace=0.42)
save(fig, "fig_sweep.pdf")

# =====================================================================
# Panels embedded in the TikZ architecture figure (window 2019-07-14 08:12, as in B1 Fig. 1)
# =====================================================================
WID = 1073
win = df[df.window_id == WID].sort_values("lead_min")
t0 = win.valid_time.iloc[0] - pd.Timedelta(minutes=2)
day = df[df.date == "2019-07-14"].drop_duplicates("valid_time").set_index("valid_time").sort_index()
hist_t = [t0 - pd.Timedelta(minutes=m) for m in range(14, -1, -2)]
hist = [float(day.observed_kw.iloc[np.argmin(np.abs((day.index - t).total_seconds()))]) for t in hist_t]
hist_gap = [float(np.min(np.abs((day.index - t).total_seconds()))) for t in hist_t]
assert max(hist_gap) <= 60, hist_gap
hmin = np.arange(-14, 1, 2)

# input: PV history, smart-persistence anchor (future) and the noisy residual
rng = np.random.default_rng(3)
fig, ax = plt.subplots(figsize=(1.30, 0.62))
ax.plot(hmin, hist, color=C_OBS, marker="o", ms=1.8, lw=0.8)
ax.plot(lead, win.spm_kw, color=C_SPM, ls="--", lw=0.9)
anchor = win.spm_kw.values
ax.scatter(lead, anchor + 1.2 * rng.standard_normal(8), s=2.5, color="#7a7a7a", zorder=3)
ax.axvline(0, color="#999999", lw=0.5)
ax.set_xlim(-15, 17)
ax.set_ylim(1.5, 8.5)
ax.set_xticks([-14, 0, 16])
ax.set_yticks([2, 5, 8])
ax.tick_params(labelsize=5.2, length=1.5, pad=1)
ax.grid(False)
save(fig, "arch_pv_in.pdf")

# output: calibrated 30-member ensemble (band 5-95 %, median) vs observed
fig, ax = plt.subplots(figsize=(1.45, 0.72))
ax.fill_between(lead, win.ddim_p05_kw, win.ddim_p95_kw, color=BAND, alpha=0.9, lw=0)
ax.plot(hmin, hist, color=C_OBS, marker="o", ms=1.6, lw=0.7)
ax.plot(lead, win.ddim_median_kw, color=C_OURS, lw=1.0)
ax.plot(lead, win.spm_kw, color=C_SPM, lw=0.7, ls="--")
ax.plot(lead, win.observed_kw, ls="none", marker="o", ms=2.2, mfc="white", mec=C_OBS, mew=0.6)
ax.axvline(0, color="#999999", lw=0.5)
ax.set_xlim(-15, 17)
ax.set_ylim(1.5, 8.5)
ax.set_xticks([-14, 0, 16])
ax.set_yticks([2, 5, 8])
ax.tick_params(labelsize=5.2, length=1.5, pad=1)
ax.grid(False)
save(fig, "arch_pv_out.pdf")

# noise tile and the generated image residual (x0 in residual space, first member, +16 min)
noise = np.clip(rng.normal(0.5, 0.28, (64, 64, 3)), 0, 1)
Image.fromarray((noise * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST).save(
    os.path.join(FIG, "arch_noise.png"))
res_img = tile("ours_typical_gen_8.png").astype(float) - tile("ours_typical_obs_0.png").astype(float)
res_vis = np.clip(128 + 4.0 * res_img, 0, 255).astype(np.uint8)
Image.fromarray(res_vis).resize((128, 128), Image.NEAREST).save(os.path.join(FIG, "arch_residual.png"))
for name in ["ours_typical_obs_0", "ours_typical_gen_1", "ours_typical_gen_4", "ours_typical_gen_8",
             "ours_typical_obs_1", "ours_typical_obs_4", "ours_typical_obs_8"]:
    Image.fromarray(tile(name + ".png")).resize((128, 128), Image.NEAREST).save(
        os.path.join(FIG, f"arch_{name}.png"))
print("history PV (kW):", np.round(hist, 2), "| window CRPS inputs ok")
