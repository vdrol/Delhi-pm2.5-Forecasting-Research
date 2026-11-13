import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from physics import mixing_height

OUT = "figures_jei"
os.makedirs(OUT, exist_ok=True)

GROUND, ERA5, ERA5PHYS, GPHYS, PERSIST, LIGHT, INK = "#d9622b", "#2a6fc4", "#7b5bc4", "#f0a882", "#555555", "#bdbdbd", "#111111"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
                     "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.7,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.15})

def panel(ax, letter):
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="left")

def save(fig, name):
    png = f"{OUT}/{name}.png"
    fig.savefig(png); plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(f"{OUT}/{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    im.save(f"{OUT}/{name}.jpg", quality=95, dpi=(300, 300))

c = pd.read_csv("data/delhi_hourly.csv", index_col=0, parse_dates=True)
c = c.join(mixing_height(c))
e = pd.read_csv("data/era5_delhi_openmeteo.csv", skiprows=3, parse_dates=["time"], index_col="time")
e.columns = ["blh", "ws100", "wd100", "cloud", "t2m"]

win = c[c.index.month.isin([11, 12, 1])]
g = win.groupby(win.index.hour)
fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.2), sharex=True, gridspec_kw=dict(hspace=0.25))
for a, col, lab in [(ax[0], "pm25", "PM2.5 (µg/m³)"), (ax[1], "h_mix", "Mixing height (m)")]:
    med, q1, q3 = g[col].median(), g[col].quantile(0.25), g[col].quantile(0.75)
    a.fill_between(med.index, q1, q3, color="#9a9a9a", alpha=0.3, lw=0)
    a.plot(med.index, med, color=INK, lw=2.2)
    a.set_ylabel(lab)
ax[1].set_xlabel("Hour of day (IST)"); ax[1].set_xticks(range(0, 24, 3))
panel(ax[0], "A"); panel(ax[1], "B")
save(fig, "Figure1")

d = c[["pm25", "vc"]].resample("D").mean().dropna()
d = d[d.index.month.isin([10, 11, 12, 1, 2])]
x, y = np.log(d.vc), np.log(d.pm25)
b, a0 = np.polyfit(x, y, 1)
fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.scatter(d.vc, d.pm25, s=12, color="#7a7a7a", alpha=0.45, lw=0)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(np.exp(xs), np.exp(a0 + b * xs), color=INK, lw=2.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Daily mean ventilation coefficient (m²/s)")
ax.set_ylabel("Daily mean PM2.5 (µg/m³)")
save(fig, "Figure2")

j = c.join(e)
w = j[j.index.month.isin([11, 12, 1])]
gm = w.groupby(w.index.hour).median()
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.6), gridspec_kw=dict(wspace=0.45))
ax[0].plot(gm.index, gm.h_mix, color=GROUND, lw=2.2, ls="--", label="Ground stations")
ax[0].plot(gm.index, gm.blh, color=ERA5, lw=2.2, label="ERA5")
ax[0].set_ylabel("Mixing height (m)")
ax[1].plot(gm.index, gm.ws, color=GROUND, lw=2.2, ls="--", label="Ground stations")
ax[1].plot(gm.index, gm.ws100, color=ERA5, lw=2.2, label="ERA5 (100 m)")
ax[1].set_ylabel("Wind speed (m/s)"); ax[1].set_ylim(0, 5)
for a in ax:
    a.set_xlabel("Hour of day (IST)"); a.set_xticks(range(0, 24, 6))
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, ["Ground stations", "ERA5"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
panel(ax[0], "A"); panel(ax[1], "B")
save(fig, "Figure3")

P = pd.read_csv("results/v2_test_predictions.csv", index_col=0, parse_dates=["target_time"])
W = P[P.target_time.dt.month.isin([10, 11, 12, 1, 2])].copy()
W["day"] = W.target_time.dt.normalize()
days = [gr for _, gr in W.groupby("day")]
models = ["N1 AI + ERA5 data", "N4 AI corrects ERA5 box model", "M2 AI only (ground sensors)",
          "N2 AI + ERA5 physics", "M3 AI + ground physics", "M0 Persistence", "N3 ERA5 box model only"]
names = {"N1 AI + ERA5 data": "AI + ERA5 data", "N4 AI corrects ERA5 box model": "AI correcting box model",
         "M2 AI only (ground sensors)": "AI, ground stations only", "N2 AI + ERA5 physics": "AI + ERA5 data + physics",
         "M3 AI + ground physics": "AI + ground-station physics", "M0 Persistence": "Persistence",
         "N3 ERA5 box model only": "Box model only"}
colors = {"N1 AI + ERA5 data": ERA5, "N4 AI corrects ERA5 box model": ERA5PHYS, "M2 AI only (ground sensors)": GROUND,
          "N2 AI + ERA5 physics": ERA5PHYS, "M3 AI + ground physics": GPHYS, "M0 Persistence": PERSIST,
          "N3 ERA5 box model only": LIGHT}
hatch = {"N4 AI corrects ERA5 box model": "//", "N2 AI + ERA5 physics": "//", "M3 AI + ground physics": "//",
         "N3 ERA5 box model only": "//"}
rm = lambda dd, m: np.sqrt(np.mean((np.exp(dd[m]) - np.exp(dd.obs)) ** 2))
point = {m: rm(W, m) for m in models}
rng = np.random.default_rng(2)
boot = {m: [] for m in models}
for _ in range(2000):
    s = pd.concat([days[i] for i in rng.integers(0, len(days), len(days))])
    for m in models:
        boot[m].append(rm(s, m))
ci = {m: (float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))) for m in models}
models = sorted(models, key=lambda m: point[m])
fig, ax = plt.subplots(figsize=(4.75, 4.2))
yy = np.arange(len(models))
vals = [point[m] for m in models]
err = np.array([[point[m] - ci[m][0] for m in models], [ci[m][1] - point[m] for m in models]])
ax.barh(yy, vals, color=[colors[m] for m in models], hatch=[hatch.get(m, "") for m in models],
        edgecolor="white", height=0.62, xerr=err, error_kw=dict(ecolor=INK, lw=1.2, capsize=4))
ax.set_yticks(yy); ax.set_yticklabels([names[m] for m in models])
for i, m in enumerate(models):
    ax.text(ci[m][1] + 1.5, i, f"{point[m]:.1f}", va="center", fontsize=12)
ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(c[1] for c in ci.values()) * 1.15)
ax.set_xlabel("Forecast error, RMSE (µg/m³)")
save(fig, "Figure4")
json.dump({names[m]: dict(rmse=point[m], lo95=ci[m][0], hi95=ci[m][1]) for m in models},
          open("results/v2_fig4_rmse_ci.json", "w"), indent=1)

a = json.load(open("results/v2_ablation.json"))
order = ["AI only (M2)", "+ boundary-layer height only", "+ cloud cover only", "+ wind speed at 100 m only",
         "+ wind direction at 100 m only", "+ wind at 100 m only", "+ all ERA5 data (N1)"]
labels = ["Ground stations only", "+ boundary-layer height", "+ cloud cover", "+ wind speed (100 m)",
          "+ wind direction (100 m)", "+ wind speed and direction", "+ all ERA5 data"]
m = [a[k][0] for k in order]; sd = [a[k][1] for k in order]
fig, ax = plt.subplots(figsize=(4.75, 4.2))
yy = np.arange(len(labels))
mk = ["s"] + ["o"] * 6
cl = [GROUND, LIGHT, LIGHT, ERA5, ERA5, ERA5, ERA5]
ax.errorbar(m, yy, xerr=sd, fmt="none", ecolor=INK, lw=1.2, capsize=4)
for i in range(len(m)):
    ax.scatter(m[i], yy[i], s=90, marker=mk[i], color=cl[i], edgecolor=INK, linewidth=0.8, zorder=3)
ax.axvline(m[0], color=INK, ls="--", lw=1)
ax.set_yticks(yy); ax.set_yticklabels(labels)
ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(63, 71)
ax.set_xlabel("Winter forecast error, RMSE (µg/m³)")
save(fig, "Figure5")

ep = P.set_index("target_time").loc["2024-11-08":"2024-11-28"]
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(ep.index, np.exp(ep.obs), color=INK, lw=2.2, label="Observed")
ax.plot(ep.index, np.exp(ep["M0 Persistence"]), color=PERSIST, lw=1.4, ls=":", label="Persistence")
ax.plot(ep.index, np.exp(ep["M2 AI only (ground sensors)"]), color=GROUND, lw=1.8, ls="--", label="AI, ground stations only")
ax.plot(ep.index, np.exp(ep["N1 AI + ERA5 data"]), color=ERA5, lw=1.8, label="AI + ERA5 data")
ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_xlabel("Date (2024)")
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=4))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "Figure6")
print("JEI figures written:", sorted(os.listdir(OUT)))
