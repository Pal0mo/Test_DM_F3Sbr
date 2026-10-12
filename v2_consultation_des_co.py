# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:30:21 2026
@author: aless / laeti / sidi
"""

import spey
import pandas as pd
import numpy as np
 
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
 
import re
import pickle
from pathlib import Path
import json
 
from scipy.interpolate import griddata

from math import inf

import time

LAM = float(input("Lambda = "))

# ──────────────────────────────────────────────────────────────────────────────
# Backend pyhf : jax + minuit (bien plus rapide que le backend numpy par défaut)
# ──────────────────────────────────────────────────────────────────────────────
import pyhf
try:
    pyhf.set_backend("jax", "minuit")
    print("[pyhf] Backend : jax + minuit")
except Exception as e:
    print(f"[pyhf] /!\\ jax/minuit indisponible ({e}), retour au backend numpy par défaut "
          f"(sera nettement plus lent sur atlas_susy_2018_22). "
          f"Installer avec : pip install jax jaxlib iminuit")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
analyses = ["atlas_susy_2018_22", "atlas_susy_2018_17", "atlas_exot_2018_06", "cms_sus_19_006", "cms_exo_20_004"]
fichiers = ["atlas_susy_2018_22_C", "atlas_susy_2018_22_GGd", "atlas_susy_2018_22_SSd", "atlas_susy_2018_17", "atlas_exot_2018_06", "cms_sus_19_006", "cms_exo_20_004"]
 
# atlas_susy_2018_22 couvre les 3 sous-analyses (C/GGd/SSd) dans un seul .info,
# distinguées par le préfixe des ids de région
ATLAS_2018_22_PREFIX = {
    "atlas_susy_2018_22_C":   "MB_C_",
    "atlas_susy_2018_22_GGd": "MB_GGd_",
    "atlas_susy_2018_22_SSd": "MB_SSd_",
}
 
# atlas_susy_2018_22 (C / GGd / SSd) : workspace pyhf (.json), un fichier par sous-région
ANALYSES_PYHF = ["atlas_susy_2018_22_C", "atlas_susy_2018_22_GGd", "atlas_susy_2018_22_SSd"]
 
# atlas_susy_2018_17 / atlas_exot_2018_06 : pas de covariance ni de JSON disponibles -> best region et covariances max et min
ANALYSES_BEST_REGION = ["atlas_susy_2018_17", "atlas_exot_2018_06"]
 
# cms_sus_19_006 : les 12 premières régions du .info sont des agrégats, on ne garde que les régions individuelles
CMS_SUS_OFFSET = 12
 
base_dir = Path("/Users/aless/Desktop/MAD/stage/Points")  # dossier des points du scan
info_dir = Path("/Users/aless/Desktop/MAD/stage")         # dossier des .info et .json
res_dir = Path(f"/Users/aless/Desktop/MAD/stage/Resultats_{LAM}") # Pour les figures et pickle avec resultats
res_dir.mkdir(parents=True, exist_ok=True)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Extraction des points du scan
# ──────────────────────────────────────────────────────────────────────────────
def extract_points():
    """ Extrait les points, i.e. les masses et la section efficace"""
    points = []
    for run_dir in base_dir.glob("mass*"):
        name = run_dir.name
        
        # Infos dans le titre du dossier
        my = float(re.search(r"mass5920005_([\d.]+)", name).group(1))
        mx = float(re.search(r"mass51_([\d.]+)", name).group(1))
        lam = float(re.search(r"dmf3d33_([\d.]+)", name).group(1))
        
        # Infos dans le fichier banner
        banner_path = run_dir / "mg5_output" / "run_01_tag_1_banner.txt"
        banner_text = banner_path.read_text()
        xs = float(re.search(r"Integrated weight \(pb\)\s*:\s*([\d.eE+-]+)", banner_text).group(1))
        
        points.append({"xsection": xs, "mx": int(mx), "my": int(my), "lambda" : float(lam)})
    return points
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Extraction des régions / covariances depuis les fichiers .info
# ──────────────────────────────────────────────────────────────────────────────
def extract(fichier):
    """Extrait la luminosité et les régions (id, nobs, nb, deltanb) depuis un .info."""
    tree = ET.parse(fichier)
    root = tree.getroot()
    lumi = float(root.find("lumi").text)
    rows = [
        {
            "id":      region.attrib["id"],
            "nobs":    float(region.find("nobs").text),
            "nb":      float(region.find("nb").text),
            "deltanb": float(region.find("deltanb").text),
        }
        for region in root.findall(".//region")
    ]
    return pd.DataFrame(rows), lumi
 
 

def extract_cov(fichier):
    """Extrait les ids de région et la matrice de covariance (sous forme de dict) depuis un .info."""
    tree = ET.parse(fichier)
    root = tree.getroot()
    ids = []
    covariances = {}
    for region in root.findall(".//region"):
        region_id = region.attrib["id"]
        ids.append(region_id)
        cov_tags = region.findall("covariance")
        if cov_tags:
            covariances[region_id] = {cov.attrib["region"]: float(cov.text) for cov in cov_tags}
    return ids, covariances
 
 
def cov_mat(ids, covariances):
    """Construit la matrice de covariance carrée à partir du dict {id: {id: cov}}."""
    matrix = np.zeros((len(ids), len(ids)))
    for i, l1 in enumerate(ids):
        for j, l2 in enumerate(ids):
            matrix[i, j] = covariances[l1][l2]
    return matrix
 
 
def build_cov_matrices():
    """Construit les matrices de covariance CMS"""
    cov_matrices = {}
 
    ids_sus, cov_sus = extract_cov(str(info_dir / "cms_sus_19_006.info"))
    cov_matrices["cms_sus_19_006"] = cov_mat(ids_sus[CMS_SUS_OFFSET:], cov_sus)
 
    ids_exo, cov_exo = extract_cov(str(info_dir / "cms_exo_20_004.info"))
    cov_matrices["cms_exo_20_004"] = cov_mat(ids_exo, cov_exo)
 
    return cov_matrices
 
 
def normalize_region_id(region_id):
    """
    Normalise un id de région pour servir de clé de merge entre le .info et le
    .saf : certaines analyses (ex: atlas_susy_2018_17) utilisent des tirets
    dans le .info ('SR-8ij50-0ib-MJ500') alors que les fichiers .saf générés
    par MA5 utilisent des underscores ('SR_8ij50_0ib_MJ500'). On uniformise
    en remplaçant '-' par '_'. Pour les analyses qui n'ont pas de tiret dans
    leurs ids (atlas_susy_2018_22, CMS, atlas_exot_2018_06), cette fonction
    est un no-op : safe à appliquer partout.
    """
    return region_id.replace("-", "_")


def load_static_regions():
    """Charge une seule fois les régions (id, nobs, nb, deltanb, analysis) et la
    luminosité pour chaque fichier d'analyse."""
    regions = {}
 
    # atlas_susy_2018_22 : un seul .info, splitté par préfixe pour C / GGd / SSd
    df_full, lumi_22 = extract(str(info_dir / "atlas_susy_2018_22.info"))
    for fichier, prefix in ATLAS_2018_22_PREFIX.items():
        df = df_full[df_full["id"].str.startswith(prefix)].reset_index(drop=True)
        df["analysis"] = "atlas_susy_2018_22"
        df["id_key"] = df["id"].map(normalize_region_id)
        regions[fichier] = (df, lumi_22)
 
    # autres analyses : un .info par fichier
    for fichier in ("atlas_susy_2018_17", "atlas_exot_2018_06", "cms_sus_19_006", "cms_exo_20_004"):
        df, lumi = extract(str(info_dir / (fichier + ".info")))
        df["analysis"] = fichier
        df["id_key"] = df["id"].map(normalize_region_id)
        regions[fichier] = (df, lumi)
 
    return regions
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Extraction des efficacités depuis les fichiers .saf (cutflows MA5)
# ──────────────────────────────────────────────────────────────────────────────
def ext_eff(analyses, point_dossier):
    """Extrait nf, n0 et calcule l'efficacité par région, pour un point du scan."""
    rows = []
    for analysis in analyses:
        cutflow_dir = (base_dir / point_dossier / "Output" / "SAF" / "defaultset" / analysis / "Cutflows")
        for saf_file in cutflow_dir.glob("*.saf"):
            region = saf_file.stem
            text = saf_file.read_text()
            # Nombre d'événements initiales
            init_block = re.search(r"<InitialCounter>(.*?)</InitialCounter>", text, re.DOTALL).group(1)
            n0 = float(re.search(r"^.*# nentries", init_block, re.MULTILINE).group(0).split()[0])
            # Nombr d'événements à l'issue des cuts
            last_counter = re.findall(r"<Counter>(.*?)</Counter>", text, re.DOTALL)[-1]
            nf = float(re.search(r"^.*# nentries", last_counter, re.MULTILINE).group(0).split()[0])
            
            rows.append({
                "id": region,
                "id_key": normalize_region_id(region),
                "analysis": analysis,
                "efficiency": nf / n0,
            })
    return pd.DataFrame(rows, columns=["id", "id_key", "analysis", "efficiency"])
 
 
# ──────────────────────────────────────────────────────────────────────────────
# pyhf : chargement des workspaces et construction du patch signal
# ──────────────────────────────────────────────────────────────────────────────
def load_pyhf_workspaces():
    """
    Charge les workspaces JSON pour les analyses dans ANALYSES_PYHF
    (atlas_susy_2018_22 : C / GGd / SSd).
    Retourne un dict { nom_analyse : {"workspace": dict, "channels": [str]} }.
    Si un .json est absent, get_stat_result lèvera une erreur explicite pour ce fichier.
    """
    pyhf_data = {}
    for analysis in ANALYSES_PYHF:
        json_path = info_dir / (analysis + ".json")
        if json_path.exists():
            with open(json_path) as fp:
                ws = json.load(fp)
            channel_names = [ch["name"] for ch in ws["channels"]]
            pyhf_data[analysis] = {"workspace": ws, "channels": channel_names}
            print(f"[pyhf] Workspace chargé : {analysis} ({len(channel_names)} canaux)")
        else:
            print(f"[pyhf] ⚠ Pas de JSON pour {analysis} — ce fichier échouera dans le scan")
    return pyhf_data
 
 
def region_id_to_channel_name(region_id, prefix):
    """
    Convertit un id de région ATLAS tel qu'il apparaît dans le .info
    (ex: 'MB_C_SR_2_1600_16') en nom de canal du workspace pyhf
    (ex: 'SR_2_1600_16_cuts') : on retire le préfixe 'MB_<sous-région>_'
    et on ajoute le suffixe '_cuts' utilisé dans les JSON fournis.
    """
    return region_id[len(prefix):] + "_cuts"
 
 
def build_signal_patch(workspace, channel_names, nsignal_dict):
    """
    Construit un patch jsonpatch qui injecte un échantillon "signal" dans les
    canaux du workspace présents dans nsignal_dict (typiquement les canaux SR
    uniquement ; les canaux de contrôle CRQ/CRT/CRW/CRY, absents de
    nsignal_dict, restent inchangés).
 
    Le nom du modifieur "normfactor" est repris directement du POI déclaré
    dans le workspace (measurements[0]["config"]["poi"], ex: "mu_SIG"), pour
    que le signal injecté soit bien piloté par le paramètre d'intérêt utilisé
    dans le fit, et non par un paramètre "orphelin" non déclaré.
 
    workspace     : dict JSON du workspace pyhf
    channel_names : liste ordonnée des noms de canaux (depuis ws["channels"])
    nsignal_dict  : dict { nom_de_canal -> nsignal } (clés = noms de canaux
                    du workspace, pas les ids du .info -> voir region_id_to_channel_name)
 
    Retour : liste de patches (format jsonpatch) prête pour spey-pyhf
    """
    poi_name = workspace["measurements"][0]["config"]["poi"]
    patch = []
    for idx, ch_name in enumerate(channel_names):
        if ch_name not in nsignal_dict:
            continue  # canal de contrôle : pas de signal à injecter
        nsig = nsignal_dict[ch_name]
        n_bins = len(workspace["channels"][idx]["samples"][0]["data"])
        patch.append({
            "op": "add",
            "path": f"/channels/{idx}/samples/{len(workspace['channels'][idx]['samples'])}",
            "value": {
                "name": "signal",
                "data": [nsig / n_bins] * n_bins,  # signal réparti uniformément sur les bins
                "modifiers": [
                    {"name": poi_name, "type": "normfactor", "data": None}
                ]
            }
        })
    return patch
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Sélection de la meilleure région (analyses sans covariance ni pyhf)
# ──────────────────────────────────────────────────────────────────────────────
def best_region(df, nsignal):
    """Sélectionne la région qui minimise mu_obs (uncorrelated_background)."""
    mu_obs = np.inf
    for i in range(len(df)):
        stat_model = spey.get_backend("default.uncorrelated_background")(
            signal_yields=[nsignal[i]],
            background_yields=[df["nb"].iloc[i]],
            data=[df["nobs"].iloc[i]],
            absolute_uncertainties=[df["deltanb"].iloc[i]],
            analysis=df["id"].iloc[i],
        )
        try:
            mu = stat_model.poi_upper_limit(expected=spey.ExpectationType.observed)
            if mu < mu_obs:
                mu_obs = mu
        except ValueError:
            continue
    return mu_obs
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Construction du modèle statistique spey et calcul de mu_obs / CLs, par fichier
# ──────────────────────────────────────────────────────────────────────────────
def _spey_result(stat_model):
    mu_obs = stat_model.poi_upper_limit(expected=spey.ExpectationType.observed)
    return mu_obs
 
 
def get_stat_result(fichier, df, nsignal, cov_matrices, pyhf_ws):
    """
    Retourne mu_obs pour un fichier donné, en choisissant le backend spey
    approprié :
      - ANALYSES_PYHF (atlas_susy_2018_22 C/GGd/SSd)         -> pyhf (workspace .json)
      - ANALYSES_BEST_REGION (atlas_susy_2018_17/exot_2018_06) -> best_region (toujours)
      - sinon, covariance connue (CMS)                        -> correlated_background
    """
  
    if fichier in ANALYSES_PYHF:
        if fichier not in pyhf_ws:
            raise ValueError(f"Workspace pyhf manquant pour {fichier} (fichier .json introuvable)")
        ws_data = pyhf_ws[fichier]["workspace"]
        ch_names = pyhf_ws[fichier]["channels"]
        prefix = ATLAS_2018_22_PREFIX[fichier]
        nsignal_dict = {
            region_id_to_channel_name(rid, prefix): nsig
            for rid, nsig in zip(df["id"].tolist(), nsignal.tolist())
        }
        patch = build_signal_patch(ws_data, ch_names, nsignal_dict)
        stat_model = spey.get_backend("pyhf")(
            analysis=fichier, background_only_model=ws_data, signal_patch=patch,
        )
        return _spey_result(stat_model)
 
    if fichier in ANALYSES_BEST_REGION:
        return best_region(df, nsignal)
 
    if fichier in cov_matrices:
        offset = CMS_SUS_OFFSET if fichier == "cms_sus_19_006" else 0
        stat_model = spey.get_backend("default.correlated_background")(
            signal_yields=nsignal.tolist()[offset:],
            background_yields=df["nb"].tolist()[offset:],
            data=df["nobs"].tolist()[offset:],
            covariance_matrix=cov_matrices[fichier].tolist(),
            analysis=fichier,
        )
        return _spey_result(stat_model)
 
    raise ValueError(f"Analyse inconnue (pas de config pyhf/best_region/covariance) : {fichier}")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Boucle principale du scan (séquentielle : un point après l'autre)
# ──────────────────────────────────────────────────────────────────────────────
def main(lam):
    t_total = time.perf_counter()

    points = extract_points()
    static_regions = load_static_regions()
    cov_matrices = build_cov_matrices()
    pyhf_ws = load_pyhf_workspaces()

    points = [p for p in points if p["lambda"] == lam]

    pickle_path = res_dir / f"resultats_scan_lam-{lam}.pkl"

    # Charger les résultats déjà calculés s'il y a un pickle existant, et ne pas retraiter les points (mx, my) déjà présents dedans.
    if pickle_path.exists():
        with open(pickle_path, "rb") as historique:
            resultats_scan = pickle.load(historique)
        deja_traites = {(p["mx"], p["my"]) for p in resultats_scan}
        print(f"[Historique] {len(resultats_scan)} points déjà présents dans {pickle_path.name}")
    else:
        resultats_scan = []
        deja_traites = set()

    points = [p for p in points if (p["mx"], p["my"]) not in deja_traites]
    print(f"{len(points)} nouveaux points à calculer.")

    for point_num, point in enumerate(points):
        t_point = time.perf_counter()

        print(f"\nPoint {point_num + 1}/{len(points)}")

        xsection, mx, my = point["xsection"], point["mx"], point["my"]

        pattern = f"mass5920005_{float(my)}_mass51_{float(mx)}_dmf3d33_{lam}_xs_*"
        matches = list(base_dir.glob(pattern))

        if not matches:
            print(f"⚠ Dossier introuvable pour mx={mx}, my={my}")
            continue

        point_dossier = matches[0].name

        t_eff = time.perf_counter()
        df_eff = ext_eff(analyses, point_dossier)
        print(f"  ext_eff : {time.perf_counter() - t_eff:.2f} s")

        resultats_par_analyse = {}

        for f in fichiers:
            t_analysis = time.perf_counter()

            print(f"  Début analyse {f}")

            df_regions, lumi = static_regions[f]
            df = df_regions.merge(
                df_eff, on=["id_key", "analysis"], how="left", suffixes=("", "_saf")
            ).drop(columns=["id_key", "id_saf"])

            n_match = df["efficiency"].notna().sum()
            if n_match == 0:
                print(f"    ⚠ Aucun matching .info/.saf pour {f} (mx={mx}, my={my}) "
                      f"— vérifier la convention de nommage des régions")

            df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce").fillna(0.0)
            nsignal = xsection * lumi * 1000 * df["efficiency"].to_numpy()
            print(f"    régions matchées: {n_match}/{len(df)} | max nsignal = {np.nanmax(nsignal):.3e}")

            try:
                mu_obs = get_stat_result(
                    f, df, nsignal, cov_matrices, pyhf_ws
                )
                rejetee = mu_obs < 1.0
            except Exception as e:
                print(f"    ✗ Erreur {f}, mx={mx}, my={my}: {e}")
                mu_obs, rejetee = np.nan, False

            resultats_par_analyse[f] = [rejetee, mu_obs]

            print(f"  Fin analyse {f} : {time.perf_counter() - t_analysis:.2f} s")

        resultats_scan.append({"mx": mx, "my": my, "resultats": resultats_par_analyse})

        print(f"Point terminé en {time.perf_counter() - t_point:.2f} s")

    print(f"\nTemps total : {(time.perf_counter() - t_total)/60:.2f} min")
    with open(pickle_path, "wb") as fh:
        pickle.dump(resultats_scan, fh)
    print(f"resultats_scan sauvegardé dans {pickle_path} ({len(resultats_scan)} points au total)")

    return resultats_scan


# ──────────────────────────────────────────────────────────────────────────────
# Interpolation de mu_obs (scan irrégulier -> champ continu)
# ──────────────────────────────────────────────────────────────────────────────
def interp_mu(resultats_scan, f, N=1000, method="cubic"):
    """
    Interpolation de mu sur une grille
    """
    
    # Extraire les resultats du scan
    mx = np.array([p["mx"] for p in resultats_scan], dtype=float)
    my = np.array([p["my"] for p in resultats_scan], dtype=float)
    mu = np.array([p["resultats"][f][1] for p in resultats_scan], dtype=float)
 
    valid = (mu > 0) & np.isfinite(mu)
    if valid.sum() < 4:
        return None, None, None
 
    mx_v, my_v, log_mu_v = mx[valid], my[valid], np.log10(mu[valid])
    
    # Créer la grille
    tab_mx = np.linspace(mx_v.min(), mx_v.max(), N)
    tab_my = np.linspace(my_v.min(), my_v.max(), N)
    grid_mx, grid_my = np.meshgrid(tab_mx, tab_my, indexing="ij")
 
    # Interpoler log(mu) 
    log_mu_interp = griddata(
        (mx_v, my_v), log_mu_v, (grid_mx, grid_my), method=method
    )
 
    log_mu_interp[grid_mx >= grid_my] = np.nan  # zone non physique 
 
    return tab_mx, tab_my, 10 ** log_mu_interp # renvoie mu et pas log(mu)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_excl(resultats_scan, lam):
    mx_arr = np.array([point["mx"] for point in resultats_scan])
    my_arr = np.array([point["my"] for point in resultats_scan])

    c = ["r", "g", "b", "orange", "m", "pink", "k", "yellow"]
 
    diag = np.linspace(min(mx_arr.min(), my_arr.min()), max(my_arr.max(), mx_arr.max()))
    
    # ── Plot 1 : mu_obs en couleur (LogNorm) ──────────────────────────────────
    fig_1, axs_1 = plt.subplots(3, 3, figsize=(15, 13))
    for i, (ax, f) in enumerate(zip(axs_1.flat, fichiers)):
        mu_array = np.array([point["resultats"][f][1] for point in resultats_scan], dtype=float)
 
        mask_valid = (mu_array > 0) & np.isfinite(mu_array)
        mask_nan = np.isnan(mu_array)
        mask_inf = np.isinf(mu_array)
 
        if i == len(fichiers) - 1:
            ax = axs_1[2, 1]
 
        if mask_valid.sum() == 0:
            ax.set_title(f + " (sans données)")
            continue
 
        sc = ax.scatter(
            my_arr[mask_valid], mx_arr[mask_valid], c=mu_array[mask_valid],
            cmap="viridis",
            norm=LogNorm(vmin=mu_array[mask_valid].min(), vmax=mu_array[mask_valid].max()),
            label=r"$\mu_{obs}$"
        )
        fig_1.colorbar(sc, ax=ax, label=r"$\log\,\mu_{obs}$")
 
        ax.scatter(my_arr[mask_nan], mx_arr[mask_nan],
                   color="red", marker="x", s=40, linewidths=1.5, label="NaN")
        ax.scatter(my_arr[mask_inf], mx_arr[mask_inf],
                   color="gold", marker="*", s=60, label="inf")
         
        # Region non-physique
        ax.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
        ax.fill_between(diag, diag, diag.max(), facecolor='grey', alpha=.5)
 
        ax.set_title(f)
        ax.set_ylabel("MX [GeV]")
        ax.set_xlabel("MY [GeV]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
 
    axs_1[2, 0].axis("off")
    axs_1[2, 2].axis("off")
    fig_1.suptitle(r"Signal strength $\mu_{obs}$ par analyse")
    plt.tight_layout()

 
    # ── Plot 2 : exclusion (exclu / non exclu) ────────────────────────────────
    fig_2, axs_2 = plt.subplots(3, 3, figsize=(15, 13))
    for i, (ax, f) in enumerate(zip(axs_2.flat, fichiers)):
        if i == len(fichiers) - 1:
            ax = axs_2[2, 1]
 
        rejetee_arr = np.array([point["resultats"][f][0] for point in resultats_scan])
 
        # Region non-physique
        ax.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
        ax.fill_between(diag, diag, diag.max(), facecolor='grey', alpha=.5)        

        ax.scatter(my_arr[rejetee_arr], mx_arr[rejetee_arr], color=c[i], label="Exclu")
        ax.scatter(my_arr[~rejetee_arr], mx_arr[~rejetee_arr], color="gray", label="Non exclu", alpha=0.4)
         
        ax.set_title(f)
        ax.set_ylabel(r"$M_X$ [GeV]")
        ax.set_xlabel(r"$M_Y$ [GeV]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.5)
 
    axs_2[2, 0].axis("off")
    axs_2[2, 2].axis("off")
    fig_2.suptitle(rf"Exclusion des paramètres d'après le LHC ($\lambda$ = {lam})")
    plt.tight_layout()
    
    
    # ── Plot 3 : mu_obs interpolé (LogNorm) ───────────────────────────────────
    fig_3, axs_3 = plt.subplots(3, 3, figsize=(15, 13))
    for i, (ax, ax_1, ax_2, f) in enumerate(zip(axs_3.flat, axs_1.flat, axs_2.flat, fichiers)):
        if i == len(fichiers) - 1:
            ax = axs_3[2, 1]
            ax_1= axs_1[2, 1]
            ax_2 = axs_2[2, 1]
 
        tab_mx, tab_my, mu_interp = interp_mu(resultats_scan, f)
        if tab_mx is None or np.all(np.isnan(mu_interp)):
            ax.set_title(f + " (sans données)")
            continue
 
        pc = ax.pcolormesh(
            tab_my, tab_mx, mu_interp,
            cmap="viridis",
            norm=LogNorm(vmin=np.nanmin(mu_interp), vmax=np.nanmax(mu_interp)),
            shading="auto",
        )
        fig_3.colorbar(pc, ax=ax, label=r"$\log\,\mu_{obs}$")
        
        # Region non-physique
        ax.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
        ax.fill_between(diag, diag, diag.max(), facecolor='grey', alpha=.5)
        
        
        # contours
        ax.contour(tab_my, tab_mx, mu_interp, levels=[1.0], colors="r", linewidths=2)
        ax.plot([], [], color="r", linewidth=3, label="Limite d'exclusion")
        
        ax_1.contour(tab_my, tab_mx, mu_interp, levels=[1.0], colors="r", linewidths=2)
        ax_1.plot([], [], color="r", linewidth=3, label="Limite d'exclusion")
        
        ax_2.contour(tab_my, tab_mx, mu_interp, levels=[1.0], colors=c[i], linewidths=2)
        ax_2.contourf(tab_my, tab_mx, mu_interp, levels=[0, 1.0], colors=[c[i]], alpha=0.35) # relleno
        ax_2.plot([], [], color=c[i], linewidth=3, label="Limite d'exclusion")
 
        ax.set_title(f)
        ax.set_ylabel("MX [GeV]")
        ax.set_xlabel("MY [GeV]")
        ax.grid(alpha=0.3)
 
    axs_3[2, 0].axis("off")
    axs_3[2, 2].axis("off")
    fig_3.suptitle(r"Signal strength $\mu_{obs}$ interpolé par analyse")
    plt.tight_layout()
    plt.show()
 
    # ── Plot 4 : exclusion interpolée (mu_obs <1) ───────────────────────────
    fig_4, axs_4 = plt.subplots(3, 3, figsize=(15, 13))
    fig_glob, ax_glob = plt.subplots(figsize=(15, 13))
    
    for i, (ax, f) in enumerate(zip(axs_4.flat, fichiers)):
        if i == len(fichiers) - 1:
            ax = axs_4[2, 1]
 
        tab_mx, tab_my, mu_interp = interp_mu(resultats_scan, f)
        if tab_mx is None or np.all(np.isnan(mu_interp)):
            ax.set_title(f + " (sans données)")
            continue
 
        # NaN -> NaN ; sinon True si mu_obs < 1 (exclu), False sinon
        exclu_interp = np.where(np.isnan(mu_interp), np.nan, mu_interp < 1.0) 
 
        # Region non-physique
        ax.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
        ax.fill_between(diag, diag, diag.max(), facecolor='grey', alpha=.5)

        
        ax.pcolormesh(
            tab_my, tab_mx, exclu_interp,
            cmap="RdGy_r", vmin=0, vmax=1, shading="auto",
        )
        ax.contour(tab_my, tab_mx, mu_interp, levels=[1.0], colors="r", linewidths=2)
        ax.plot([], [], color="r", linewidth=4, label="Limite d'exclusion")
        
        # Pour le plot global 
        ax_glob.contour(tab_my, tab_mx, mu_interp, levels=[1.0], colors=c[i], linewidths=2)
        ax_glob.contourf(tab_my, tab_mx, mu_interp, levels=[0, 1.0], colors=[c[i]], alpha=0.35) # relleno
        ax_glob.plot([], [], color=c[i], linewidth=3, label=f)
        
        # Pour les subplots
        ax.set_title(f)
        ax.set_ylabel(r"$M_X$ [GeV]")
        ax.set_xlabel(r"$M_Y$ [GeV]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
 
    axs_4[2, 0].axis("off")
    axs_4[2, 2].axis("off")
    fig_4.suptitle(rf"Exclusion interpolée ($\mu_{{obs}} < 1$) :  $\lambda$ = {lam}")
    plt.tight_layout()
    
    
    # Pour la figure gloable
        # Region non-physique
    ax_glob.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
    ax_glob.fill_between(diag, diag, diag.max(), facecolor='grey', alpha=.5)
    
    ax_glob.set_title(rf"Limites d'exclusion par analyse ($\lambda$ = {lam}")
    ax_glob.set_xlabel(r"$M_Y$ [GeV]")
    ax_glob.set_ylabel(r"$M_X$ [GeV]")
    ax_glob.legend(fontsize=8)
    ax_glob.grid(alpha=0.3)
    
    
    fig_glob.savefig(res_dir / "lims_exclusion.png", dpi=300, bbox_inches="tight")
    fig_1.savefig(res_dir / "mu_obs_par_analyse.png", dpi=300, bbox_inches="tight")
    fig_2.savefig(res_dir / "exclusion_par_analyse.png", dpi=300, bbox_inches="tight")
    fig_3.savefig(res_dir / "mu_obs_interp_par_analyse.png", dpi=300, bbox_inches="tight") 
    fig_4.savefig(res_dir / "exclusion_interp_par_analyse.png", dpi=300, bbox_inches="tight") 
    plt.show()


resultats_scan = main(lam=LAM)
plot_excl(resultats_scan, lam = LAM)

"""
resultats_scan = [{'mx': 25,
  'my': 600,
  'resultats': {'atlas_susy_2018_22_C': [np.True_,
    np.float64(0.3676576854239616)],
   'atlas_susy_2018_22_GGd': [np.True_, np.float64(0.0664524059293613)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.08585295552390894)],
   'atlas_susy_2018_17': [np.False_, np.float64(23.122089969689647)],
   'atlas_exot_2018_06': [np.True_, np.float64(0.18356601603005196)],
   'cms_sus_19_006': [np.True_, np.float64(0.01806807284467775)],
   'cms_exo_20_004': [np.False_, np.float64(2.8094804413862686)]}},
 {'mx': 400,
  'my': 1000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(2.523016617346213)],
   'atlas_susy_2018_22_GGd': [np.True_, np.float64(0.8691724092006403)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.6000447362978423)],
   'atlas_susy_2018_17': [np.False_, np.float64(341.22330468703905)],
   'atlas_exot_2018_06': [np.False_, np.float64(1.7386412204521489)],
   'cms_sus_19_006': [np.True_, np.float64(0.12293343808066562)],
   'cms_exo_20_004': [np.False_, np.float64(23.10539081811101)]}},
 {'mx': 1595,
  'my': 1600,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(51014.30670752129)],
   'atlas_susy_2018_22_GGd': [np.False_, inf],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(53888.3898167842)],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [np.False_, np.float64(26218.198637767895)],
   'cms_sus_19_006': [np.False_, np.float64(57703.45637084135)],
   'cms_exo_20_004': [np.False_, inf]}},
 {'mx': 195,
  'my': 200,
  'resultats': {'atlas_susy_2018_22_C': [np.True_,
    np.float64(0.7495605637340659)],
   'atlas_susy_2018_22_GGd': [np.True_, np.float64(0.2739501543066185)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.44936665080689575)],
   'atlas_susy_2018_17': [np.False_, np.float64(74.53383267060315)],
   'atlas_exot_2018_06': [np.True_, np.float64(0.6346649657964419)],
   'cms_sus_19_006': [np.True_, np.float64(0.3789561565215478)],
   'cms_exo_20_004': [np.False_, np.float64(6.236107072045381)]}},
 {'mx': 400,
  'my': 600,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(1.3975746611566193)],
   'atlas_susy_2018_22_GGd': [np.True_, np.float64(0.22234620456249)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.9210591424450405)],
   'atlas_susy_2018_17': [np.False_, np.float64(125.28947352707405)],
   'atlas_exot_2018_06': [np.False_, np.float64(1.6966425557398794)],
   'cms_sus_19_006': [np.True_, np.float64(0.10590563019763126)],
   'cms_exo_20_004': [np.False_, np.float64(58.29986773764068)]}},
 {'mx': 1400,
  'my': 1800,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(628.3087200297282)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(97.9169495213075)],
   'atlas_susy_2018_22_SSd': [False, np.float64(103.70974451684395)],
   'atlas_susy_2018_17': [np.False_, np.float64(18235.27498604326)],
   'atlas_exot_2018_06': [np.False_, np.float64(392.54056624595995)],
   'cms_sus_19_006': [np.False_, np.float64(26.511844949348355)],
   'cms_exo_20_004': [np.False_, np.float64(7726.087040774111)]}},
 {'mx': 25,
  'my': 1400,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(2.150097715864365)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(4.113427991308274)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.8886358379817088)],
   'atlas_susy_2018_17': [np.False_, np.float64(1007.9015139279545)],
   'atlas_exot_2018_06': [np.False_, np.float64(1.716007259937082)],
   'cms_sus_19_006': [np.True_, np.float64(0.35830232907771814)],
   'cms_exo_20_004': [np.False_, np.float64(25.72860524726196)]}},
 {'mx': 800,
  'my': 1400,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(24.126320152733463)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(7.1461058499186745)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(4.98974207610076)],
   'atlas_susy_2018_17': [np.False_, np.float64(1870.7085273395892)],
   'atlas_exot_2018_06': [np.False_, np.float64(11.211792880462795)],
   'cms_sus_19_006': [np.False_, np.float64(1.1138182584819025)],
   'cms_exo_20_004': [np.False_, np.float64(205.96140129708755)]}},
 {'mx': 650,
  'my': 2800,
  'resultats': {'atlas_susy_2018_22_C': [False, np.float64(388.5541867392148)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(1579.20026157593)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(177.68186602068965)],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [np.False_, np.float64(393.7991524906762)],
   'cms_sus_19_006': [np.False_, np.float64(103.59081845601433)],
   'cms_exo_20_004': [np.False_, np.float64(3964.5868493020157)]}},
 {'mx': 1200,
  'my': 1600,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(219.73988348541417)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(36.75817058885062)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(37.6092113876838)],
   'atlas_susy_2018_17': [np.False_, np.float64(17768.688324284092)],
   'atlas_exot_2018_06': [np.False_, np.float64(132.7833261201178)],
   'cms_sus_19_006': [np.False_, np.float64(9.01277422316748)],
   'cms_exo_20_004': [np.False_, np.float64(2691.256641733738)]}},
 {'mx': 25,
  'my': 1000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(1.2819841537030625)],
   'atlas_susy_2018_22_GGd': [np.True_, np.float64(0.6172114245669704)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.32035883536796234)],
   'atlas_susy_2018_17': [np.False_, np.float64(181.97459768002707)],
   'atlas_exot_2018_06': [np.True_, np.float64(0.5601166272317728)],
   'cms_sus_19_006': [np.True_, np.float64(0.07077410826022634)],
   'cms_exo_20_004': [np.False_, np.float64(9.148147775374369)]}},
 {'mx': 595,
  'my': 600,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(109.71218564517588)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(127.32576056064067)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(71.85831178622675)],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [np.False_, np.float64(70.65507769996782)],
   'cms_sus_19_006': [np.False_, np.float64(79.38860190336914)],
   'cms_exo_20_004': [np.False_, np.float64(3333.928272312597)]}},
 {'mx': 1600,
  'my': 1800,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(1085.3031163293642)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(254.3310652894353)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(777.1947689326877)],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [np.False_, np.float64(2394.330888257318)],
   'cms_sus_19_006': [np.False_, np.float64(121.28975798644623)],
   'cms_exo_20_004': [np.False_, np.float64(62081.48820346175)]}},
 {'mx': 1195,
  'my': 1200,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(3665.9475025879865)],
   'atlas_susy_2018_22_GGd': [False, np.float64(7000.234662875615)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(3234.6004282514805)],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [np.False_, np.float64(2091.0052610149464)],
   'cms_sus_19_006': [False, np.float64(3745.7533711292626)],
   'cms_exo_20_004': [np.False_, inf]}},
 {'mx': 200,
  'my': 600,
  'resultats': {'atlas_susy_2018_22_C': [np.True_,
    np.float64(0.7239464488154439)],
   'atlas_susy_2018_22_GGd': [np.True_, np.float64(0.07834975441291596)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.16717661501933773)],
   'atlas_susy_2018_17': [np.False_, np.float64(25.1451976362371)],
   'atlas_exot_2018_06': [np.True_, np.float64(0.33062890192716893)],
   'cms_sus_19_006': [np.True_, np.float64(0.02832242034996835)],
   'cms_exo_20_004': [np.False_, np.float64(2.5766158118096154)]}},
 {'mx': 400,
  'my': 1400,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(3.595848917475964)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(4.734843712355502)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(1.3906181128183768)],
   'atlas_susy_2018_17': [np.False_, np.float64(1318.995096727781)],
   'atlas_exot_2018_06': [np.False_, np.float64(3.5605189265441606)],
   'cms_sus_19_006': [np.True_, np.float64(0.45600859036889674)],
   'cms_exo_20_004': [np.False_, np.float64(34.97176193440395)]}},
 {'mx': 600,
  'my': 1800,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(9.924908787038197)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(30.181285815341713)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(4.5468810811221605)],
   'atlas_susy_2018_17': [np.False_, np.float64(8297.999502924364)],
   'atlas_exot_2018_06': [np.False_, np.float64(8.41162966679016)],
   'cms_sus_19_006': [np.False_, np.float64(2.475253705427315)],
   'cms_exo_20_004': [np.False_, np.float64(184.36383490519341)]}},
 {'mx': 995,
  'my': 1000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(1025.925229732909)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(1806.4706049159154)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(925.8455563762838)],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [np.False_, np.float64(615.0665685893106)],
   'cms_sus_19_006': [np.False_, np.float64(1208.4928837651719)],
   'cms_exo_20_004': [np.False_, np.float64(38387.25703446381)]}},
 {'mx': 1000,
  'my': 2000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(42.53639383438107)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(101.98874262617356)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(18.367552668453474)],
   'atlas_susy_2018_17': [np.False_, np.float64(31517.747758405254)],
   'atlas_exot_2018_06': [np.False_, np.float64(33.64267801538999)],
   'cms_sus_19_006': [np.False_, np.float64(8.867591517324986)],
   'cms_exo_20_004': [np.False_, np.float64(716.2135822604127)]}},
 {'mx': 1795,
  'my': 1800,
  'resultats': {'atlas_susy_2018_22_C': [np.False_, inf],
   'atlas_susy_2018_22_GGd': [np.False_, inf],
   'atlas_susy_2018_22_SSd': [np.False_, inf],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [False, inf],
   'cms_sus_19_006': [np.False_, inf],
   'cms_exo_20_004': [np.False_, inf]}},
 {'mx': 25,
  'my': 1800,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(5.994188675104365)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(21.54139309702294)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(2.76800133558844)],
   'atlas_susy_2018_17': [np.False_, np.float64(6234.240936319809)],
   'atlas_exot_2018_06': [np.False_, np.float64(5.63122477942352)],
   'cms_sus_19_006': [np.False_, np.float64(1.5777901036245277)],
   'cms_exo_20_004': [np.False_, np.float64(101.78018921232783)]}},
 {'mx': 650,
  'my': 1400,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(10.586441911384576)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(6.081060429562175)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(2.911429425064357)],
   'atlas_susy_2018_17': [np.False_, np.float64(1826.3273553585987)],
   'atlas_exot_2018_06': [np.False_, np.float64(4.777979413130097)],
   'cms_sus_19_006': [np.True_, np.float64(0.6951582370904592)],
   'cms_exo_20_004': [np.False_, np.float64(77.16448692804644)]}},
 {'mx': 1395,
  'my': 1400,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(11411.349547455522)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(27070.579570803973)],
   'atlas_susy_2018_22_SSd': [False, np.float64(10932.274775058973)],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [np.False_, np.float64(6684.781865077184)],
   'cms_sus_19_006': [np.False_, np.float64(12569.52469585172)],
   'cms_exo_20_004': [np.False_, inf]}},
 {'mx': 200,
  'my': 2000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(12.96877123416247)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(50.78369988527888)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(6.045177581414734)],
   'atlas_susy_2018_17': [np.False_, np.float64(17018.41372599864)],
   'atlas_exot_2018_06': [np.False_, np.float64(12.848119696832295)],
   'cms_sus_19_006': [np.False_, np.float64(3.451694522534666)],
   'cms_exo_20_004': [np.False_, np.float64(213.74892139836697)]}},
 {'mx': 200,
  'my': 1200,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(1.9154632901765347)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(1.8159792766043938)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.6485931362571771)],
   'atlas_susy_2018_17': [np.False_, np.float64(397.3185430110907)],
   'atlas_exot_2018_06': [np.False_, np.float64(1.2270085537527735)],
   'cms_sus_19_006': [np.True_, np.float64(0.16828759061758664)],
   'cms_exo_20_004': [np.False_, np.float64(16.348206930358373)]}},
 {'mx': 800,
  'my': 1600,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(18.30722641293061)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(15.611987170746074)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(6.298517852367223)],
   'atlas_susy_2018_17': [np.False_, np.float64(3467.7427966418827)],
   'atlas_exot_2018_06': [np.False_, np.float64(13.130870805363159)],
   'cms_sus_19_006': [np.False_, np.float64(1.7225613323955773)],
   'cms_exo_20_004': [np.False_, np.float64(181.9927098238473)]}},
 {'mx': 400,
  'my': 1800,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(7.325742737444337)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(25.885109006894517)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(3.387184278544197)],
   'atlas_susy_2018_17': [np.False_, np.float64(7371.525021279074)],
   'atlas_exot_2018_06': [np.False_, np.float64(6.622466901907741)],
   'cms_sus_19_006': [np.False_, np.float64(1.9140824206344895)],
   'cms_exo_20_004': [np.False_, np.float64(126.99620451371753)]}},
 {'mx': 25,
  'my': 1200,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(1.5658361050888372)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(1.617115125728571)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.5617046047521124)],
   'atlas_susy_2018_17': [np.False_, np.float64(411.8296264478379)],
   'atlas_exot_2018_06': [np.False_, np.float64(1.2023489237954519)],
   'cms_sus_19_006': [np.True_, np.float64(0.15536274033790476)],
   'cms_exo_20_004': [np.False_, np.float64(13.545067212541682)]}},
 {'mx': 1200,
  'my': 2000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(116.52913609129004)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(127.3815344980169)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(41.88174372448897)],
   'atlas_susy_2018_17': [np.False_, np.float64(44289.26348138707)],
   'atlas_exot_2018_06': [np.False_, np.float64(110.02101838892554)],
   'cms_sus_19_006': [np.False_, np.float64(13.383008431995659)],
   'cms_exo_20_004': [np.False_, np.float64(1183.441810154673)]}},
 {'mx': 25,
  'my': 800,
  'resultats': {'atlas_susy_2018_22_C': [np.True_,
    np.float64(0.6830638638689102)],
   'atlas_susy_2018_22_GGd': [np.True_, np.float64(0.20979590075373672)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.16211870374819842)],
   'atlas_susy_2018_17': [np.False_, np.float64(75.27633212015186)],
   'atlas_exot_2018_06': [np.True_, np.float64(0.458493571749244)],
   'cms_sus_19_006': [np.True_, np.float64(0.0316783731716148)],
   'cms_exo_20_004': [np.False_, np.float64(6.5642357651811345)]}},
 {'mx': 600,
  'my': 1000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(8.045030892624554)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(1.1384212402850364)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(1.4776785675959183)],
   'atlas_susy_2018_17': [np.False_, np.float64(238.3251399786763)],
   'atlas_exot_2018_06': [np.False_, np.float64(3.926739839505336)],
   'cms_sus_19_006': [np.True_, np.float64(0.33171852708954025)],
   'cms_exo_20_004': [np.False_, np.float64(72.76360425232575)]}},
 {'mx': 1400,
  'my': 2000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(644.8719972224225)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(181.4315338747661)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(106.64804077836871)],
   'atlas_susy_2018_17': [np.False_, np.float64(52336.561183868274)],
   'atlas_exot_2018_06': [np.False_, np.float64(174.38977921631073)],
   'cms_sus_19_006': [np.False_, np.float64(24.60715221566914)],
   'cms_exo_20_004': [np.False_, np.float64(4281.994612428155)]}},
 {'mx': 600,
  'my': 1200,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(8.405004495323144)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(2.5508002310714453)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(1.7495338447162916)],
   'atlas_susy_2018_17': [np.False_, np.float64(813.4733299011993)],
   'atlas_exot_2018_06': [np.False_, np.float64(3.9750605885217305)],
   'cms_sus_19_006': [np.True_, np.float64(0.38198303285173124)],
   'cms_exo_20_004': [np.False_, np.float64(72.75482845351783)]}},
 {'mx': 1000,
  'my': 1600,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(76.93571977238426)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(22.00027985405059)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(13.971063606660902)],
   'atlas_susy_2018_17': [np.False_, np.float64(10103.326663318949)],
   'atlas_exot_2018_06': [np.False_, np.float64(26.68923426892841)],
   'cms_sus_19_006': [np.False_, np.float64(3.269685528335527)],
   'cms_exo_20_004': [np.False_, np.float64(603.3398035475159)]}},
 {'mx': 800,
  'my': 1000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(12.9831632213936)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(3.0957866046461353)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(10.047173074342428)],
   'atlas_susy_2018_17': [np.False_, np.float64(946.5455572075678)],
   'atlas_exot_2018_06': [np.False_, np.float64(32.42739068520465)],
   'cms_sus_19_006': [np.False_, np.float64(1.3860899306138843)],
   'cms_exo_20_004': [np.False_, np.float64(708.8879790731307)]}},
 {'mx': 800,
  'my': 2000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(23.61935443764146)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(77.70567364510995)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(10.84887348604112)],
   'atlas_susy_2018_17': [np.False_, np.float64(23939.45099990361)],
   'atlas_exot_2018_06': [np.False_, np.float64(21.03505662456665)],
   'cms_sus_19_006': [np.False_, np.float64(6.211526764178196)],
   'cms_exo_20_004': [np.False_, np.float64(432.0241720363978)]}},
 {'mx': 1000,
  'my': 1200,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(40.09052753353976)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(9.434295975346243)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(33.63098080675011)],
   'atlas_susy_2018_17': [np.False_, np.float64(3483.787088865559)],
   'atlas_exot_2018_06': [np.False_, np.float64(102.05408462013266)],
   'cms_sus_19_006': [np.False_, np.float64(4.219403236214024)],
   'cms_exo_20_004': [np.False_, np.float64(2044.323111399711)]}},
 {'mx': 1000,
  'my': 1400,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(74.21826301145316)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(11.770642854882691)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(12.738795863472593)],
   'atlas_susy_2018_17': [np.False_, np.float64(7309.4634509388015)],
   'atlas_exot_2018_06': [np.False_, np.float64(43.566976940025)],
   'cms_sus_19_006': [np.False_, np.float64(3.091429804295609)],
   'cms_exo_20_004': [np.False_, np.float64(968.4662885272041)]}},
 {'mx': 25,
  'my': 1600,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(3.212579638224277)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(9.546543685704428)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(1.4706876698603033)],
   'atlas_susy_2018_17': [np.False_, np.float64(2506.2175231743868)],
   'atlas_exot_2018_06': [np.False_, np.float64(2.76222316167303)],
   'cms_sus_19_006': [np.True_, np.float64(0.7764730591939342)],
   'cms_exo_20_004': [np.False_, np.float64(52.26288056416491)]}},
 {'mx': 25,
  'my': 2000,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(12.860970928296071)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(49.17579391109115)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(6.0127942060644575)],
   'atlas_susy_2018_17': [np.False_, np.float64(17186.300796840544)],
   'atlas_exot_2018_06': [np.False_, np.float64(12.523707574764128)],
   'cms_sus_19_006': [np.False_, np.float64(3.349961070182653)],
   'cms_exo_20_004': [np.False_, np.float64(195.3176179060696)]}},
 {'mx': 800,
  'my': 1200,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(23.98394448651765)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(4.040304695244611)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(4.546162889200685)],
   'atlas_susy_2018_17': [np.False_, np.float64(1978.8191458698511)],
   'atlas_exot_2018_06': [np.False_, np.float64(13.799431035151844)],
   'cms_sus_19_006': [np.False_, np.float64(1.0049808798117572)],
   'cms_exo_20_004': [np.False_, np.float64(229.20757183718797)]}},
 {'mx': 600,
  'my': 1400,
  'resultats': {'atlas_susy_2018_22_C': [np.False_,
    np.float64(7.759781300675999)],
   'atlas_susy_2018_22_GGd': [np.False_, np.float64(5.682042430368574)],
   'atlas_susy_2018_22_SSd': [np.False_, np.float64(2.4544574075688175)],
   'atlas_susy_2018_17': [np.False_, np.float64(1609.2500653210054)],
   'atlas_exot_2018_06': [np.False_, np.float64(4.520448397768815)],
   'cms_sus_19_006': [np.True_, np.float64(0.6267524616699616)],
   'cms_exo_20_004': [np.False_, np.float64(62.28425962681081)]}},
 {'mx': 200,
  'my': 400,
  'resultats': {'atlas_susy_2018_22_C': [np.True_,
    np.float64(0.45211904722244295)],
   'atlas_susy_2018_22_GGd': [np.True_, np.float64(0.08184669264392386)],
   'atlas_susy_2018_22_SSd': [np.True_, np.float64(0.23332693209055583)],
   'atlas_susy_2018_17': [False, inf],
   'atlas_exot_2018_06': [np.True_, np.float64(0.27947301133202473)],
   'cms_sus_19_006': [np.True_, np.float64(0.029211413509615897)],
   'cms_exo_20_004': [np.False_, np.float64(2.2887897185548858)]}}]

plot_excl(resultats_scan)
"""