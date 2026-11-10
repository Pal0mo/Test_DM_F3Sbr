#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:33:22 2026

@author: aless
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from final_consultation_des_co import interp_mu, lumi_tag_str, fmt_lumi


# ──────────────────────────────────────────────────────────────────────────────
# Luminosités cibles (doivent être identiques à celles utilisées lors du scan,
# cf. vfinal_consultation_des_co.py
# ──────────────────────────────────────────────────────────────────────────────
LUMI_CIBLE = {"3": {"atlas": 332, "cms": 326.03}, "HL": {"atlas": 3000, "cms": 3000}}


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des pickles Run 2 / Run 3 / HL-LHC
# ──────────────────────────────────────────────────────────────────────────────
def import_res(res_dir_run2, res_dir_run3, res_dir_hllhc, lam, lumi_run3=None, lumi_hllhc=None):
    
    """
    Charge les pickles `resultats_scan` et `lumis` du Run 2,  Run 3 et HL-LHC
    """
    lumi_run3 = lumi_run3 if lumi_run3 is not None else LUMI_CIBLE["3"]
    lumi_hllhc = lumi_hllhc if lumi_hllhc is not None else LUMI_CIBLE["HL"]

    res_dir_run2 = Path(res_dir_run2)
    res_dir_run3 = Path(res_dir_run3)
    res_dir_hllhc = Path(res_dir_hllhc)

    path2 = res_dir_run2 / f"resultats_scan_lam-{lam}_run2.pkl"
    path3 = res_dir_run3 / f"resultats_scan_lam-{lam}_run3-{lumi_tag_str(lumi_run3)}.pkl"
    pathhl = res_dir_hllhc / f"resultats_scan_lam-{lam}_hllhc-{lumi_tag_str(lumi_hllhc)}.pkl"

    lumis_path2 = path2.with_name(path2.name.replace("resultats_scan_", "lumis_"))
    lumis_path3 = path3.with_name(path3.name.replace("resultats_scan_", "lumis_"))
    lumis_pathhl = pathhl.with_name(pathhl.name.replace("resultats_scan_", "lumis_"))

    with open(path2, "rb") as f:
        res2 = pickle.load(f)
    with open(path3, "rb") as f:
        res3 = pickle.load(f)
    with open(pathhl, "rb") as f:
        reshl = pickle.load(f)

    with open(lumis_path2, "rb") as f:
        lumis2 = pickle.load(f)
    with open(lumis_path3, "rb") as f:
        lumis3 = pickle.load(f)
    with open(lumis_pathhl, "rb") as f:
        lumishl = pickle.load(f)

    return res2, res3, reshl, lumis2, lumis3, lumishl


# ──────────────────────────────────────────────────────────────────────────────
# Outils pour le tracé
# ──────────────────────────────────────────────────────────────────────────────
def grille_commune(res2, res3, reshl, N=1000):
    """Construit une grille (mx, my) et la diagonale non-physique"""
    tous = list(res2) + list(res3) + list(reshl)
    mx_arr = np.array([p["mx"] for p in tous], dtype=float)
    my_arr = np.array([p["my"] for p in tous], dtype=float)

    vals_mx = np.linspace(mx_arr.min(), mx_arr.max(), N)
    vals_my = np.linspace(my_arr.min(), my_arr.max(), N)
    diag = np.linspace(min(mx_arr.min(), my_arr.min()), max(my_arr.max(), mx_arr.max()))

    return vals_mx, vals_my, diag


def zone_non_physique(ax, diag):
    """ Trace la zone non-physique"""
    ax.plot(diag, diag, "--", color="gray", linewidth=1, zorder=1)
    ax.fill_between(diag, diag, diag.max(), edgecolor="k", alpha=0.75, facecolor="k", zorder=1)


def mu_min_global(resultats_scan, fichiers, vals_mx, vals_my):
    """Interpole mu par analyse puis renvoie le minimum """
    interps = [interp_mu(resultats_scan, f, vals_mx, vals_my) for f in fichiers]
    return np.nanmin(np.stack(interps), axis=0)

c = [
    "#00C853",  # vert
    "#FF6D00",  # orange
    "#FF1744",  # rouge
    "#2979FF",  # bleu
    "#AA00FF",  # violet
    "#FFD600",  # jaune
    "#00B8D4",  # cyan
]

# Couleurs et styles par run (utilisés partout où les 3 runs sont superposés)
color_run2 = "blue"
color_run3 = "red"
color_hllhc = "darkorange"

style_run2 = "-"
style_run3 = "--"
style_hllhc = ":"


# ──────────────────────────────────────────────────────────────────────────────
# Tracé
# ──────────────────────────────────────────────────────────────────────────────
def plot_comp(res2, res3, reshl, lam, lumis2, lumis3, lumishl, comp_dir, N=500):

    vals_mx, vals_my, diag = grille_commune(res2, res3, reshl, N=N)

    # ── Plot 1 : comparaison par analyse (grille 3x3, comme plot_excl) ────────
    fig_1, axs_1 = plt.subplots(3, 3, figsize=(16, 9))
    for i, (ax, f) in enumerate(zip(axs_1.flat, fichiers)):
        if i == len(fichiers) - 1:
            ax = axs_1[2, 1]

        # Interpolation des mu
        mu2 = interp_mu(res2, f, vals_mx, vals_my)
        mu3 = interp_mu(res3, f, vals_mx, vals_my)
        muhl = interp_mu(reshl, f, vals_mx, vals_my)
        
        # enlever l'erreur d'interpolation present comme des lignes en dehors de la region d'exclusion 
        # Coordonnées de la grille : axe horizontal = M_Y, axe vertical = M_X
        MY, MX = np.meshgrid(vals_my, vals_mx)
        
        # Zone contenant les deux contours isolés
        machin = (MY > 1900) & (MX > 1000) & (MX < 1600)
        if lam == 1:
            mu2  = np.ma.masked_where(machin, mu2)
            mu3  = np.ma.masked_where(machin, mu3)
            muhl = np.ma.masked_where(machin, muhl)

        zone_non_physique(ax, diag) # tracé de la zone non-physique

        ax.contour(vals_my, vals_mx, mu2, levels=[1.0], colors=color_run2, linestyles=style_run2, linewidths=2)
        ax.contour(vals_my, vals_mx, mu3, levels=[1.0], colors=color_run3, linestyles=style_run3, linewidths=2)
        ax.contour(vals_my, vals_mx, muhl, levels=[1.0], colors=color_hllhc, linestyles=style_hllhc, linewidths=2)

        ax.set_title(
            f
            + f"\nRun 2: {fmt_lumi(lumis2, f)} " + r"$fb^{-1}$"
            + f"  |  Run 3: {fmt_lumi(lumis3, f)} " + r"$fb^{-1}$"
            + f"  |  HL-LHC: {fmt_lumi(lumishl, f)} " + r"$fb^{-1}$"
        )
        ax.set_xlabel(r"$M_Y$ [GeV]")
        ax.set_ylabel(r"$M_X$ [GeV]")
        ax.grid(alpha=0.3)

        legend_lignes = [
            Line2D([], [], color=color_run2, linewidth=2, linestyle=style_run2, label="Run 2"),
            Line2D([], [], color=color_run3, linewidth=2, linestyle=style_run3, label="Run 3"),
            Line2D([], [], color=color_hllhc, linewidth=2, linestyle=style_hllhc, label="HL-LHC"),
        ]
        ax.legend(handles=legend_lignes, fontsize=8)

    axs_1[2, 0].axis("off")
    axs_1[2, 2].axis("off")
    fig_1.suptitle(
        rf"Comparaison des limites d'exclusion par analyse — Run 2 vs Run 3 vs HL-LHC ($\lambda = {lam}$)"
    )
    fig_1.tight_layout()

    # ── Plot 2 : toutes les analyses superposées ─────────────────────
    fig_2, ax = plt.subplots(figsize=(16, 9))
    zone_non_physique(ax, diag)

    legend_analyses = []

    for i, f in enumerate(fichiers):

        mu2 = interp_mu(res2, f, vals_mx, vals_my)
        mu3 = interp_mu(res3, f, vals_mx, vals_my)
        muhl = interp_mu(reshl, f, vals_mx, vals_my)
        
        # enlever l'erreur d'interpolation present comme des lignes en dehors de la region d'exclusion 
        MY, MX = np.meshgrid(vals_my, vals_mx)
        machin = (MY > 1900) & (MX > 1000) & (MX < 1600)
        
        if lam == 1:
            mu2  = np.ma.masked_where(machin, mu2)
            mu3  = np.ma.masked_where(machin, mu3)
            muhl = np.ma.masked_where(machin, muhl)

        ax.contour(vals_my, vals_mx, mu2, levels=[1.0], colors=c[i], linestyles=style_run2, linewidths=2)
        ax.contour(vals_my, vals_mx, mu3, levels=[1.0], colors=c[i], linestyles=style_run3, linewidths=2)
        ax.contour(vals_my, vals_mx, muhl, levels=[1.0], colors=c[i], linestyles=style_hllhc, linewidths=2)

        legend_analyses.append(Line2D([], [], color=c[i], linewidth=2, label=f))

    legend_runs = [
        Line2D([], [], color="black", linewidth=2, linestyle=style_run2, label="Run 2"),
        Line2D([], [], color="black", linewidth=2, linestyle=style_run3, label="Run 3"),
        Line2D([], [], color="black", linewidth=2, linestyle=style_hllhc, label="HL-LHC"),
    ]

    leg1 = ax.legend(handles=legend_analyses, fontsize=8, loc="upper left", title="Analyse", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=legend_runs, fontsize=9, loc="upper right", title="Run")

    ax.set_xlabel(r"$M_Y$ [GeV]")
    ax.set_ylabel(r"$M_X$ [GeV]")
    ax.set_title("Limites d'exclusion par analyse — Run 2 vs Run 3 vs HL-LHC \n"
                 rf"($\lambda = {lam}$)" + "\n"
                 f"Run 2: {fmt_lumi(lumis2)} " + r"$fb^{-1}$"
                 + f"  |  Run 3: {fmt_lumi(lumis3)} " + r"$fb^{-1}$"
                 + f"  |  HL-LHC: {fmt_lumi(lumishl)} " + r"$fb^{-1}$"
                 )
    ax.grid(alpha=0.3)
    fig_2.tight_layout()

    # ── Plot 3 : exclusion globale ───────────────────────────────────
    mu_min_2 = mu_min_global(res2, fichiers, vals_mx, vals_my)
    mu_min_3 = mu_min_global(res3, fichiers, vals_mx, vals_my)
    mu_min_hl = mu_min_global(reshl, fichiers, vals_mx, vals_my)
    
    # enlever l'erreur d'interpolation present comme des lignes en dehors de la region d'exclusion 
    MY, MX = np.meshgrid(vals_my, vals_mx)
    machin = (MY > 1900) & (MX > 1000) & (MX < 1600)
    
    if lam == 1:
        mu_min_2 = np.ma.masked_where(machin, mu_min_2)
        mu_min_3 = np.ma.masked_where(machin, mu_min_3)
        mu_min_hl = np.ma.masked_where(machin, mu_min_hl)
    
    fig_3, ax = plt.subplots(figsize=(14, 30))
    
    zone_non_physique(ax, diag)

    ax.contourf(vals_my, vals_mx, mu_min_2, levels=[0, 1.0], colors=[color_run2], alpha=0.35)
    ax.contour(vals_my, vals_mx, mu_min_2, levels=[1.0], colors=color_run2, linestyles=style_run2, linewidths=2.5)

    ax.contourf(vals_my, vals_mx, mu_min_3, levels=[0, 1.0], colors=[color_run3], alpha=0.35)
    ax.contour(vals_my, vals_mx, mu_min_3, levels=[1.0], colors=color_run3, linestyles=style_run3, linewidths=2.5)

    ax.contourf(vals_my, vals_mx, mu_min_hl, levels=[0, 1.0], colors=[color_hllhc], alpha=0.35)
    ax.contour(vals_my, vals_mx, mu_min_hl, levels=[1.0], colors=color_hllhc, linestyles=style_hllhc, linewidths=2.5)

    legend_lignes_plt3 = [
        Line2D([], [], color=color_run2, linewidth=8, alpha=0.5,
               label="Run 2"),
        Line2D([], [], color=color_run3, linewidth=8, alpha=0.5,
               label="Run 3"),
        Line2D([], [], color=color_hllhc, linewidth=8, alpha=0.5,
               label="HL-LHC"),
    ]
    ax.legend(
        handles=legend_lignes_plt3,
        fontsize=21,
        loc="upper right",
        framealpha=0.9,
        handlelength=2.5,
        labelspacing=0.6
    )
    
    ax.set_xlabel(r"$M_Y$ [GeV]", fontsize=24)
    ax.set_ylabel(r"$M_X$ [GeV]", fontsize=24)
    ax.tick_params(axis="both", labelsize=19)
    ax.set_title("Région d'exclusion globale : Run 2 vs Run 3 vs HL-LHC \n"
                 rf"$\lambda = {lam}$" + "\n"
                 f"Run 2: {fmt_lumi(lumis2)} " + r"$fb^{-1}$"
                 + f"  |  Run 3: {fmt_lumi(lumis3)} " + r"$fb^{-1}$"
                 + f"  |  HL-LHC: {fmt_lumi(lumishl)} " + r"$fb^{-1}$",
                 fontsize=22, pad=18)
    ax.grid(alpha=0.3)
    fig_3.tight_layout()
    
    # Aligner les limites gauche et supérieure sur la zone grise
    for ax_subplot in axs_1.flat:
        if ax_subplot.axison:
            ax_subplot.set_xlim(left=diag.min())
            ax_subplot.set_ylim(top=diag.max())

            if lam == 1:
                ax_subplot.set_xlim(right=2100)

    for fig in (fig_2, fig_3):
        ax_figure = fig.axes[0]
        ax_figure.set_xlim(left=diag.min())
        ax_figure.set_ylim(top=diag.max())

        if lam == 1:
            ax_figure.set_xlim(right=2100)

    # ── Sauvegarde ───────────────────────────────────────────────────
    comp_dir.mkdir(parents=True, exist_ok=True)

    fig_1.savefig(comp_dir / "comparaison_run2_run3_hllhc_par_analyse.png", dpi=300, bbox_inches="tight")
    fig_2.savefig(comp_dir / "comparaison_run2_run3_hllhc_superpose.png", dpi=300, bbox_inches="tight")
    fig_3.savefig(comp_dir / "comparaison_run2_run3_hllhc_globale.png", dpi=300, bbox_inches="tight")
    print(f"Sauveguardé à {comp_dir}")
    plt.show()

    return


if __name__ == "__main__":

    fichiers = ["atlas_susy_2018_22_C", "atlas_susy_2018_22_GGd", "atlas_susy_2018_22_SSd", "atlas_susy_2018_17", "atlas_exot_2018_06", "cms_sus_19_006", "cms_exo_20_004"]

    LAM = float(input("Lambda = "))

    # Où sauvegarder les résultats
    comp_dir = Path(f"/Users/aless/Desktop/MAD/stage/Resultats/Comparaison/Resultats_{LAM}_comparaison")

    lumi_run3 = LUMI_CIBLE["3"]
    lumi_hllhc = LUMI_CIBLE["HL"]

    res2, res3, reshl, lumis2, lumis3, lumishl = import_res(
        res_dir_run2=Path(f"/Users/aless/Desktop/MAD/stage/Resultats/Run_2/Resultats_{LAM}_run2"),
        res_dir_run3=Path(f"/Users/aless/Desktop/MAD/stage/Resultats/Run_3/Resultats_{LAM}_run3_{lumi_tag_str(lumi_run3)}"),
        res_dir_hllhc=Path(f"/Users/aless/Desktop/MAD/stage/Resultats/HL-LHC/Resultats_{LAM}_hllhc_{lumi_tag_str(lumi_hllhc)}"),
        lam=LAM,
        lumi_run3=lumi_run3,
        lumi_hllhc=lumi_hllhc,
    )
    plot_comp(res2, res3, reshl, LAM, lumis2, lumis3, lumishl, comp_dir, N=500)