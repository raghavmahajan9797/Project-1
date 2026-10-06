"""
Paper figures for the fsQCA study (AI capability and supply chain resilience,
35 semiconductor firms, 2023-2025).

Run from the repository root:
    python make_figures.py

Inputs
    fsQCA Cross Check Sim/raw_dataset.csv        raw 0-4 construct scores (105 firm-years)
    Final/1.Data_and_Calibration.xlsx            only for the data-status flags (Figure 1)
Outputs
    figures/Fig1 ... Fig7 (.png at 300 dpi and .pdf)

Every number plotted is recomputed here from the raw scores with the same rules
as RCodeCheck_Final.R (direct calibration, 0.5 -> 0.501, consistency >= 0.80,
PRI >= 0.50, n >= 2, >= 2 distinct firms). The intermediate solution formulas
are the ones reported by R; they are written out in SOLUTIONS below.
"""
import itertools, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

RAW = os.path.join("fsQCA Cross Check Sim", "raw_dataset.csv")
XLSX = os.path.join("Final", "1.Data_and_Calibration.xlsx")
OUT = "figures"

CONDS = ["AICAP", "AIGOV", "SUSSRC", "DIGVIS", "SUPCOLL", "SUPDIV"]
VARS = CONDS + ["SCR"]
LABEL = {"AICAP": "AI capability", "AIGOV": "AI governance", "SUSSRC": "Sustainable sourcing",
         "DIGVIS": "Digital visibility", "SUPCOLL": "Supplier collaboration",
         "SUPDIV": "Supplier diversification", "SCR": "Supply chain resilience"}
PCTL = {"AICAP": (0.500, 2.03, 4.000), "AIGOV": (0.470, 1.14, 3.144), "SUSSRC": (0.962, 2.85, 4.000),
        "DIGVIS": (0.694, 1.90, 3.470), "SUPCOLL": (0.942, 2.36, 4.000), "SUPDIV": (0.794, 1.93, 4.000),
        "SCR": (0.960, 2.43, 3.756)}
RUBRIC = {v: (1.0, 2.0, 3.0) for v in VARS}
INCL_CUT, PRI_CUT, N_CUT, FIRM_CUT = 0.80, 0.50, 2, 2

# Intermediate solutions reported by R (QCA package). "~" = absence.
# core = condition also present in the parsimonious solution.
SOLUTIONS = {
    ("Rubric", "SCR"):      [{"SUPCOLL": "C"}, {"AICAP": "C", "~SUSSRC": "C"}],
    ("Percentile", "SCR"):  [{"SUSSRC": "P", "SUPCOLL": "C"}, {"SUPCOLL": "C", "SUPDIV": "P"}],
    ("Rubric", "~SCR"):     [{"~AICAP": "C", "~AIGOV": "P", "~DIGVIS": "P", "~SUPCOLL": "C", "~SUPDIV": "C"}],
    ("Percentile", "~SCR"): [{"~AIGOV": "P", "~SUPCOLL": "C", "~SUPDIV": "P"},
                             {"~DIGVIS": "P", "~SUPCOLL": "C", "~SUPDIV": "P"},
                             {"~AICAP": "P", "~AIGOV": "P", "~DIGVIS": "P", "~SUPCOLL": "C"}],
}

SHORT = {"Advanced Micro Devices": "AMD", "Taiwan Semiconductor Manufacturing Co.": "TSMC",
         "United Microelectronics Corp.": "UMC", "ASE Technology Holding": "ASE", "Analog Devices": "ADI",
         "Applied Materials": "AMAT", "Broadcom Inc.": "Broadcom", "Intel Corporation": "Intel",
         "KLA Corporation": "KLA", "Lam Research": "Lam", "Marvell Technology": "Marvell",
         "MediaTek Inc.": "MediaTek", "Microchip Technology": "Microchip", "Micron Technology": "Micron",
         "NVIDIA Corporation": "NVIDIA", "Nanya Technology": "Nanya", "Novatek Microelectronics": "Novatek",
         "Phison Electronics": "Phison", "Qualcomm Incorporated": "Qualcomm", "Realtek Semiconductor": "Realtek",
         "Samsung Electronics": "Samsung", "Skyworks Solutions": "Skyworks", "Texas Instruments": "TI",
         "Venture Corporation": "Venture", "Himax Technologies": "Himax", "AEM Holdings": "AEM",
         "Frencken Group": "Frencken", "UMS Holdings": "UMS", "GlobalFoundries": "GlobalFoundries"}

# ---------------------------------------------------------------- style
INK, MID, LIGHT, GRID = "#1f2933", "#52606d", "#9aa5b1", "#d9dee4"
C_RUB, C_PCT = "#1b4965", "#d1603d"          # rubric (primary), percentile (robustness)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.5,
    "axes.edgecolor": MID, "axes.linewidth": 0.7, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MID, "ytick.color": MID, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fsQCA maths
def calibrate(x, e, c, i):
    """Ragin's direct method (log-odds), full membership = 0.95."""
    L = np.log(0.95 / 0.05)
    x = np.asarray(x, float)
    z = np.where(x >= c, L * (x - c) / (i - c), L * (x - c) / (c - e))
    return 1.0 / (1.0 + np.exp(-z))


def fuzzy_table(raw, anchors):
    fz = pd.DataFrame(index=raw.index)
    for v in VARS:
        m = np.round(calibrate(raw[v], *anchors[v]), 6)
        m[m == 0.5] = 0.501
        fz[v] = m
    return fz


def term_membership(fz, term):
    cols = [(1 - fz[k[1:]]) if k.startswith("~") else fz[k] for k in term]
    return np.minimum.reduce([c.values for c in cols])


def fit(x, y):
    """sufficiency consistency, PRI, coverage of X for outcome Y"""
    xy = np.minimum(x, y).sum()
    xyny = np.minimum(np.minimum(x, y), 1 - y).sum()
    return xy / x.sum(), (xy - xyny) / (x.sum() - xyny), xy / y.sum()


def necessity(x, y):
    """necessity consistency and relevance of necessity (RoN)"""
    return np.minimum(x, y).sum() / y.sum(), (1 - x).sum() / (1 - np.minimum(x, y)).sum()


def truth_table(fz, raw, y):
    rows = []
    for k, combo in enumerate(itertools.product([0, 1], repeat=len(CONDS)), start=1):
        m = np.minimum.reduce([fz[c].values if b else 1 - fz[c].values for c, b in zip(CONDS, combo)])
        members = m > 0.5
        n = int(members.sum())
        if n == 0:
            continue
        incl, pri, _ = fit(m, y)
        firms = raw.loc[members, "Company"].nunique()
        passes = incl >= INCL_CUT and pri >= PRI_CUT
        if n < N_CUT:
            status = "remainder"            # too few firm-years: treated as a remainder
        elif passes and firms >= FIRM_CUT:
            status = "retained"             # OUT = 1
        elif passes:
            status = "remainder"            # passes the cut-offs but rests on one firm
        else:
            status = "rejected"             # OUT = 0
        rows.append(dict(row=k, n=n, firms=firms, incl=incl, pri=pri, status=status,
                         one_firm=bool(n >= N_CUT and passes and firms < FIRM_CUT)))
    return pd.DataFrame(rows)


def solution_stats(fz, y, terms):
    mem = [term_membership(fz, t) for t in terms]
    sol = np.maximum.reduce(mem)
    s_incl, s_pri, s_cov = fit(sol, y)
    out = []
    for j, m in enumerate(mem):
        incl, pri, cov = fit(m, y)
        rest = [mm for jj, mm in enumerate(mem) if jj != j]
        ucov = (np.minimum(sol, y).sum() - np.minimum(np.maximum.reduce(rest), y).sum()) / y.sum() if rest else np.nan
        out.append(dict(incl=incl, pri=pri, cov=cov, ucov=ucov))
    return out, dict(incl=s_incl, pri=s_pri, cov=s_cov), sol


# ---------------------------------------------------------------- figures
def fig1_data_basis(status):
    order = sorted(VARS, key=lambda v: (status[v] == "Simulated").mean())
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    left = np.zeros(len(order))
    for lab, col in [("Observed", C_RUB), ("Estimated", "#7ea3bb"), ("Simulated", "#dfe5ea")]:
        w = np.array([(status[v] == lab).sum() for v in order], float)
        ax.barh([LABEL[v] for v in order], w, left=left, color=col, edgecolor="white", linewidth=0.8, label=lab, height=0.68)
        for yi, (wi, li) in enumerate(zip(w, left)):
            if wi >= 6:
                ax.text(li + wi / 2, yi, f"{int(wi)}", ha="center", va="center", fontsize=7.5,
                        color="white" if lab == "Observed" else INK)
        left += w
    ax.set_xlim(0, 105); ax.set_xlabel("Firm-year cells (of 105 per construct)")
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.6)
    save(fig, "Fig1_data_basis")


def fig2_calibration(raw, fzs):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharey=True)
    xs = np.linspace(0, 4, 400)
    for ax, v in zip(axes.flat, VARS):
        for name, anchors, col, ls in [("Rubric", RUBRIC, C_RUB, "-"), ("Percentile", PCTL, C_PCT, "--")]:
            ax.plot(xs, calibrate(xs, *anchors[v]), color=col, lw=1.4, ls=ls)
        ax.axhline(0.5, color=GRID, lw=0.7, zorder=0)
        ax.plot(raw[v], np.full(len(raw), -0.075), "|", color=MID, ms=5, mew=0.6, alpha=0.55, clip_on=False)
        ax.set_title(LABEL[v], fontsize=8.5); ax.set_xlim(0, 4); ax.set_ylim(-0.12, 1.02)
        ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0, 0.5, 1])
    for ax in axes[:, 0]:
        ax.set_ylabel("Set membership")
    for ax in axes[1, :3]:
        ax.set_xlabel("Raw score (0–4)")
    for ax in axes[0, :]:
        ax.set_xlabel("")
    axes[0, 3].set_xlabel("Raw score (0–4)"); axes[0, 3].tick_params(labelbottom=True)
    lg = axes[1, 3]; lg.axis("off")
    lg.legend(handles=[Line2D([], [], color=C_RUB, lw=1.4, label="Rubric anchors (1 / 2 / 3)"),
                       Line2D([], [], color=C_PCT, lw=1.4, ls="--", label="Percentile anchors\n(5th / 50th / 95th)"),
                       Line2D([], [], color=MID, marker="|", ms=6, mew=0.8, lw=0, label="Firm-year raw scores")],
              loc="center left", handlelength=2.2, labelspacing=0.9)
    fig.tight_layout(w_pad=0.8, h_pad=1.2)
    save(fig, "Fig2_calibration_curves")


def fig3_necessity(fzs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    names = [c for c in CONDS] + ["~" + c for c in CONDS]
    ypos = np.arange(len(names))[::-1]
    for ax, outcome in zip(axes, ["SCR", "~SCR"]):
        ax.axvline(0.90, color=INK, lw=0.8, ls=(0, (4, 3)))
        ax.text(0.912, len(names) - 0.08, "0.90", fontsize=7.5, color=INK, va="top")
        for name, col, mk, dy in [("Rubric", C_RUB, "o", 0.14), ("Percentile", C_PCT, "s", -0.14)]:
            fz = fzs[name]
            y = fz["SCR"].values if outcome == "SCR" else 1 - fz["SCR"].values
            for yy, cn in zip(ypos, names):
                x = (1 - fz[cn[1:]].values) if cn.startswith("~") else fz[cn].values
                cons, ron = necessity(x, y)
                ax.plot([0, cons], [yy + dy] * 2, color=col, lw=0.6, alpha=0.35)
                ax.plot(cons, yy + dy, mk, color=col, ms=4.6, mec="white", mew=0.5)
                if cons >= 0.90:
                    ax.text(0.885, yy + dy, f"RoN {ron:.2f}", ha="right", va="center", fontsize=7, color=col,
                            bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=5)
        ax.axhline(5.5, color=GRID, lw=0.7)
        ax.set_xlim(0, 1.02); ax.set_ylim(-0.6, len(names) + 0.1)
        ax.set_xlabel("Necessity consistency")
        ax.set_title("Outcome: high resilience (SCR)" if outcome == "SCR" else "Outcome: low resilience (~SCR)")
        ax.set_yticks(ypos); ax.set_yticklabels(names); ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.5)
    fig.legend(handles=[Line2D([], [], color=C_RUB, marker="o", lw=0, ms=5, label="Rubric calibration"),
                        Line2D([], [], color=C_PCT, marker="s", lw=0, ms=5, label="Percentile calibration")],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, handletextpad=0.2, columnspacing=2.0)
    save(fig, "Fig3_necessity")


def fig4_truth_table(tts):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, name in zip(axes, ["Rubric", "Percentile"]):
        tt = tts[name]
        ax.axvspan(INCL_CUT, 1.02, ymin=(PRI_CUT + 0.03) / 1.06, color="#eef3f6", zorder=0)
        ax.axvline(INCL_CUT, color=INK, lw=0.8, ls=(0, (4, 3))); ax.axhline(PRI_CUT, color=INK, lw=0.8, ls=(0, (4, 3)))
        for st, kw in [("rejected", dict(facecolor="white", edgecolor=LIGHT, linewidths=1.0)),
                       ("remainder", dict(facecolor="white", edgecolor=C_PCT, linewidths=1.0)),
                       ("retained", dict(facecolor=C_RUB, edgecolor="white", linewidths=0.6))]:
            d = tt[tt.status == st]
            ax.scatter(d.incl, d.pri, s=14 + 9 * d.n, zorder=3, **kw)
        ax.set_xlim(0.40, 1.02); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Raw consistency"); ax.set_title(f"{name} calibration")
        ax.text(0.03, 0.97, f"{(tt.status == 'retained').sum()} of {len(tt)} observed rows retained",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MID)
    axes[0].set_ylabel("PRI")
    fig.tight_layout(w_pad=1.5)
    fig.legend(handles=[Line2D([], [], marker="o", lw=0, ms=6, mfc=C_RUB, mec="white", label="Retained (OUT = 1)"),
                        Line2D([], [], marker="o", lw=0, ms=6, mfc="white", mec=LIGHT, label="Fails consistency or PRI (OUT = 0)"),
                        Line2D([], [], marker="o", lw=0, ms=6, mfc="white", mec=C_PCT, label="Fewer than 2 firm-years or 2 firms (remainder)")],
               loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, handletextpad=0.2, columnspacing=1.6)
    fig.text(0.5, -0.115, "Each bubble is one observed truth-table row (outcome SCR); bubble area grows with the number of firm-years.",
             ha="center", fontsize=7.5, color=MID)
    save(fig, "Fig4_truth_table_rows")


def fig5_configurations(stats):
    groups = [("High resilience (SCR)", [("Rubric", "SCR"), ("Percentile", "SCR")]),
              ("Low resilience (~SCR)", [("Rubric", "~SCR"), ("Percentile", "~SCR")])]
    cols = []
    for gname, keys in groups:
        for key in keys:
            for j, term in enumerate(SOLUTIONS[key]):
                cols.append((gname, key, j, term))
    ncol = len(cols)
    stat_rows = [("incl", "Consistency"), ("pri", "PRI"), ("cov", "Raw coverage"), ("ucov", "Unique coverage")]
    sol_rows = [("incl", "Solution consistency"), ("cov", "Solution coverage")]
    x0, cw, rh = 2.55, 0.86, 0.36
    nrows = 3 + len(CONDS) + len(stat_rows) + len(sol_rows)
    W, H = x0 + ncol * cw + 0.1, nrows * rh + 0.62
    fig, ax = plt.subplots(figsize=(W, H)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    xc = lambda j: x0 + (j + 0.5) * cw
    y = H - 0.25

    def span(text, j0, j1, yy, bold=False, line=True, size=8.5):
        ax.text((xc(j0) + xc(j1)) / 2, yy, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal")
        if line:
            ax.plot([xc(j0) - cw * 0.42, xc(j1) + cw * 0.42], [yy - rh * 0.46] * 2, color=MID, lw=0.7)

    ax.plot([0.05, W - 0.05], [y + rh * 0.45] * 2, color=INK, lw=1.1)
    j = 0
    for gname, keys in groups:                                   # outcome row
        n = sum(len(SOLUTIONS[k]) for k in keys); span(gname, j, j + n - 1, y, bold=True); j += n
    y -= rh; j = 0
    for gname, keys in groups:                                   # calibration row
        for k in keys:
            n = len(SOLUTIONS[k]); span(k[0], j, j + n - 1, y); j += n
    y -= rh
    counters = {}
    for j, (gname, key, tj, term) in enumerate(cols):            # path labels
        pre = "H" if key[1] == "SCR" else "L"
        counters[pre] = counters.get(pre, 0) + 1
        ax.text(xc(j), y, f"{pre}{counters[pre]}", ha="center", va="center", fontsize=8.5)
    ax.plot([0.05, W - 0.05], [y - rh * 0.5] * 2, color=INK, lw=0.7)
    y -= rh
    R_CORE, R_PER = 0.115, 0.062
    for c in CONDS:                                              # condition rows
        ax.text(0.1, y, LABEL[c], ha="left", va="center")
        ax.text(x0 - 0.12, y, c, ha="right", va="center", fontsize=7, color=LIGHT)
        for j, (_, _, _, term) in enumerate(cols):
            for k, kind in term.items():
                if k.lstrip("~") != c:
                    continue
                r = R_CORE if kind == "C" else R_PER
                if k.startswith("~"):
                    ax.add_patch(Circle((xc(j), y), r, facecolor="white", edgecolor=INK, lw=1.0))
                    d = r * 0.7071
                    ax.plot([xc(j) - d, xc(j) + d], [y - d, y + d], color=INK, lw=1.0)
                    ax.plot([xc(j) - d, xc(j) + d], [y + d, y - d], color=INK, lw=1.0)
                else:
                    ax.add_patch(Circle((xc(j), y), r, facecolor=INK, edgecolor=INK, lw=1.0))
        y -= rh
    ax.plot([0.05, W - 0.05], [y + rh * 0.5] * 2, color=INK, lw=0.7)
    for key_, lab in stat_rows:                                  # path statistics
        ax.text(0.1, y, lab, ha="left", va="center")
        for j, (_, key, tj, _) in enumerate(cols):
            v = stats[key][0][tj][key_]
            ax.text(xc(j), y, "–" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=8)
        y -= rh
    ax.plot([0.05, W - 0.05], [y + rh * 0.5] * 2, color=GRID, lw=0.7)
    for key_, lab in sol_rows:                                   # solution statistics
        ax.text(0.1, y, lab, ha="left", va="center")
        j = 0
        for gname, keys in groups:
            for k in keys:
                n = len(SOLUTIONS[k]); span(f"{stats[k][1][key_]:.3f}", j, j + n - 1, y, line=False, size=8); j += n
        y -= rh
    ax.plot([0.05, W - 0.05], [y + rh * 0.5] * 2, color=INK, lw=1.1)
    yl = y - 0.12; xl = 0.25                                     # legend
    for kind, neg, lab in [("C", False, "Core condition present"), ("P", False, "Peripheral condition present"),
                           ("C", True, "Core condition absent"), ("P", True, "Peripheral condition absent")]:
        r = R_CORE if kind == "C" else R_PER
        if neg:
            ax.add_patch(Circle((xl, yl), r, facecolor="white", edgecolor=INK, lw=1.0)); d = r * 0.7071
            ax.plot([xl - d, xl + d], [yl - d, yl + d], color=INK, lw=1.0); ax.plot([xl - d, xl + d], [yl + d, yl - d], color=INK, lw=1.0)
        else:
            ax.add_patch(Circle((xl, yl), r, facecolor=INK, edgecolor=INK))
        ax.text(xl + 0.17, yl, lab, ha="left", va="center", fontsize=7.5, color=MID)
        xl += 0.17 + 0.062 * len(lab) + 0.3
    ax.set_aspect("equal")
    save(fig, "Fig5_configuration_chart")


def fig6_xy(raw, fzs, stats):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75))
    formula = {"Rubric": "SUPCOLL + AICAP*~SUSSRC", "Percentile": "SUSSRC*SUPCOLL + SUPCOLL*SUPDIV"}
    for ax, name, col in zip(axes, ["Rubric", "Percentile"], [C_RUB, C_PCT]):
        y = fzs[name]["SCR"].values
        _, s, sol = stats[(name, "SCR")]
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8)
        ax.axvline(0.5, color=GRID, lw=0.7, zorder=0); ax.axhline(0.5, color=GRID, lw=0.7, zorder=0)
        dev = (sol > 0.5) & (y < 0.5)
        ax.scatter(sol[~dev], y[~dev], s=16, color=col, alpha=0.6, linewidths=0, zorder=3)
        ax.scatter(sol[dev], y[dev], s=20, facecolor="white", edgecolor=col, linewidths=1.1, zorder=4)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(f"Membership in solution\n{formula[name]}")
        ax.set_title(f"{name} calibration")
        ax.text(0.98, 0.03, f"Consistency {s['incl']:.3f}, coverage {s['cov']:.3f}\n"
                            f"{int(dev.sum())} deviant cases (in solution, not in SCR)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MID, linespacing=1.35)
    axes[0].set_ylabel("Membership in high resilience (SCR)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig6_xy_plots")


def fig7_case_heatmap(raw, fz):
    d = fz.copy(); d["Company"] = raw["Company"].values
    g = d.groupby("Company")[VARS].mean().sort_values("SCR", ascending=False)
    names = [SHORT.get(n, n.split()[0] + (" " + n.split()[1] if n.split()[0] in ("DB", "LX", "SK", "Hana") else "")) for n in g.index]
    fig, ax = plt.subplots(figsize=(4.6, 7.0))
    im = ax.imshow(g.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(VARS))); ax.set_xticklabels(VARS, rotation=40, ha="left", fontsize=7.5)
    ax.xaxis.tick_top(); ax.tick_params(axis="both", length=0)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5)
    ax.axvline(len(CONDS) - 0.5, color="white", lw=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(VARS)), minor=True); ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.02, aspect=30)
    cb.set_label("Mean set membership, 2023–2025 (rubric calibration)", fontsize=7.5); cb.outline.set_visible(False)
    cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(length=0, labelsize=7.5)
    save(fig, "Fig7_firm_membership_heatmap")


# ---------------------------------------------------------------- main
def main():
    raw = pd.read_csv(RAW)
    fzs = {"Rubric": fuzzy_table(raw, RUBRIC), "Percentile": fuzzy_table(raw, PCTL)}
    tts = {n: truth_table(fz, raw, fz["SCR"].values) for n, fz in fzs.items()}
    stats = {}
    for (name, outcome), terms in SOLUTIONS.items():
        fz = fzs[name]; y = fz["SCR"].values if outcome == "SCR" else 1 - fz["SCR"].values
        stats[(name, outcome)] = solution_stats(fz, y, terms)

    print("Check against RCodeCheck_Results.txt")
    for name in ["Rubric", "Percentile"]:
        tt = tts[name]
        print(f"  {name}: observed rows {len(tt)}, retained {(tt.status == 'retained').sum()}, "
              f"one-firm {sorted(tt[tt.one_firm].row)}")
    for k, (paths, sol, _) in stats.items():
        print(f"  {k}: solution incl {sol['incl']:.3f} PRI {sol['pri']:.3f} cov {sol['cov']:.3f} | "
              + "; ".join(f"{p['incl']:.3f}/{p['pri']:.3f}/{p['cov']:.3f}/{p['ucov']:.3f}" for p in paths))

    if os.path.exists(XLSX):
        st = pd.read_excel(XLSX, sheet_name="Raw Data (0-4 Scores)")
        fig1_data_basis({v: st[v + "_Status"] for v in VARS})
    else:
        print("  (workbook not found - Figure 1 skipped)")
    fig2_calibration(raw, fzs)
    fig3_necessity(fzs)
    fig4_truth_table(tts)
    fig5_configurations(stats)
    fig6_xy(raw, fzs, stats)
    fig7_case_heatmap(raw, fzs["Rubric"])
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
