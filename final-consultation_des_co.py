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
from matplotlib.colors import ListedColormap

 
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
    """Tracé des différents plots pour le travail"""
    
    # Variables globales
    mx_arr = np.array([point["mx"] for point in resultats_scan])
    my_arr = np.array([point["my"] for point in resultats_scan])

    c = ["r", "g", "b", "orange", "m", "pink", "k", "yellow"]
 
    diag = np.linspace(min(mx_arr.min(), my_arr.min()), max(my_arr.max(), mx_arr.max()))
    
    atlas_susy_2018_22 = ["atlas_susy_2018_22_C", "atlas_susy_2018_22_GGd", "atlas_susy_2018_22_SSd"]

    
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
            my_arr[mask_valid], mx_arr[mask_valid], c=mu_array[mask_valid], s=10,
            cmap="viridis",
            norm=LogNorm(vmin=mu_array[mask_valid].min(), vmax=mu_array[mask_valid].max()),
            label=r"$\mu_{obs}$"
        )
        fig_1.colorbar(sc, ax=ax, label=r"$\log\,\mu_{obs}$")
 
        ax.scatter(my_arr[mask_nan], mx_arr[mask_nan], color="red", marker="x", s=10, linewidths=1.5, label="NaN")
        ax.scatter(my_arr[mask_inf], mx_arr[mask_inf], color="gold", marker="*", s=10, label="inf")
         
        # Region non-physique
        ax.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
        ax.fill_between(diag, diag, diag.max(), edgecolor='k', alpha=0.75, facecolor="k")
 
        ax.set_title(f)
        ax.set_ylabel("MX [GeV]")
        ax.set_xlabel("MY [GeV]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
 
    axs_1[2, 0].axis("off")
    axs_1[2, 2].axis("off")
    fig_1.suptitle(r"Signal strength $\mu_{obs}$ par analyse" + rf" ($\lambda = {lam}$)")
    plt.tight_layout()

 
    # ── Plot 2 : exclusion (exclu / non exclu) ────────────────────────────────
    fig_2, axs_2 = plt.subplots(3, 3, figsize=(15, 13))
    for i, (ax, f) in enumerate(zip(axs_2.flat, fichiers)):
        if i == len(fichiers) - 1:
            ax = axs_2[2, 1]
 
        rejetee_arr = np.array([point["resultats"][f][0] for point in resultats_scan])
 
        # Region non-physique
        ax.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
        ax.fill_between(diag, diag, diag.max(), edgecolor='k', alpha=0.75, facecolor="k")        

        ax.scatter(my_arr[rejetee_arr], mx_arr[rejetee_arr], color=c[i], label="Exclu", s=10)
        ax.scatter(my_arr[~rejetee_arr], mx_arr[~rejetee_arr], color="gray", label="Non exclu", alpha=0.4, s=10)
         
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
        ax.fill_between(diag, diag, diag.max(), edgecolor='k',alpha=0.75, facecolor="k")
        
        
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
    fig_3.suptitle(r"Signal strength $\mu_{obs}$ par analyse" + rf" ($\lambda = {lam}$)")
    plt.tight_layout()
    plt.show()
 
    # ── Plot 4 : exclusion interpolée (mu_obs <1) ───────────────────────────
    fig_4, axs_4 = plt.subplots(3, 3, figsize=(15, 13))
    fig_glob, ax_glob = plt.subplots(figsize=(15, 13))
    
    mu_interp_subset_list = []
    mu_interp_all_list = []
    
    for i, (ax, f) in enumerate(zip(axs_4.flat, fichiers)):
        if i == len(fichiers) - 1:
            ax = axs_4[2, 1]
 
        tab_mx, tab_my, mu_interp = interp_mu(resultats_scan, f)
        if tab_mx is None or np.all(np.isnan(mu_interp)):
            ax.set_title(f + " (sans données)")
            continue
 
        exclu_interp = np.where(np.isnan(mu_interp), np.nan, mu_interp < 1.0) 
 
        ax.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
        ax.fill_between(diag, diag, diag.max(), edgecolor='k',alpha=0.75, facecolor="k")


        cmap_original = plt.get_cmap("RdGy_r")
        rg = cmap_original(1.0)
        
        cmap_excl = ListedColormap(["white", rg])   # 0 -> blamche (non exclu), 1 -> rouge (exclu)
        ax.pcolormesh(tab_my, tab_mx, exclu_interp, cmap=cmap_excl, vmin=0, vmax=1, shading="auto")
        ax.contour(tab_my, tab_mx, mu_interp, levels=[1.0], colors="r", linewidths=2)
        ax.plot([], [], color="r", linewidth=4, label="Limite d'exclusion")
        
        ax_glob.contour(tab_my, tab_mx, mu_interp, levels=[1.0], colors=c[i], linewidths=2)
        ax_glob.contourf(tab_my, tab_mx, mu_interp, levels=[0, 1.0], colors=[c[i]], alpha=0.35)
        ax_glob.plot([], [], color=c[i], linewidth=3, label=f)

        if f in atlas_susy_2018_22:
            mu_interp_subset_list.append(mu_interp)
        mu_interp_all_list.append(mu_interp)
            
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
    ax_glob.fill_between(diag, diag, diag.max(), edgecolor='k', alpha=0.75, facecolor="k")        
    ax_glob.set_title(rf"Limites d'exclusion par analyse ($\lambda$ = {lam})")    
    ax_glob.set_xlabel(r"$M_Y$ [GeV]")    
    ax_glob.set_ylabel(r"$M_X$ [GeV]")    
    ax_glob.legend(fontsize=8)    
    ax_glob.grid(alpha=0.3)
    
     
    # ── Plot 5 : analyse la plus contraignante (subset atlas_susy_2018_22) ───────────────    
    f_mu_min_arr = []
    excluded_arr = []
    for point in resultats_scan:
        p = point["resultats"]
        mu_min = 1
        f_min = None
        for f in atlas_susy_2018_22:
            if p[f][1] < mu_min:
                mu_min = p[f][1]
                f_min = f
        f_mu_min_arr.append(f_min)          # une entrée par point
        excluded_arr.append(f_min is not None) # analyse exlcu ou pas
    
    f_mu_min_arr = np.array(f_mu_min_arr, dtype=object)
    excluded_arr = np.array(excluded_arr)

    
    fig_5, axs_5 = plt.subplots(figsize=(15, 13))
    for i, f in enumerate(atlas_susy_2018_22):
        mask = (f_mu_min_arr == f) & excluded_arr
        axs_5.scatter(my_arr[mask], mx_arr[mask], s=10, color=c[i], label=f)
    axs_5.scatter(my_arr[~excluded_arr], mx_arr[~excluded_arr], s=10, color="grey")
    
    # Region non-physique
    axs_5.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
    axs_5.fill_between(diag, diag, diag.max(), edgecolor='k',alpha=0.75, facecolor="k")
    
    axs_5.set_title("Analyse la plus contraignante et limites d'exclusion complémentaires \n atlas_susy_2018_22")
    axs_5.set_ylabel(r"$M_X$ [GeV]")
    axs_5.set_xlabel(r"$M_Y$ [GeV]")
    axs_5.legend(fontsize=8)
    axs_5.grid(alpha=0.3)
    axs_5.plot([], [], color="k", linewidth=3, label="Région exclue")  
    
    # ── Plot 6 : analyse la plus contraignante (toutes analyses) ─────────────────────────
    f_mu_min_arr = []
    excluded_arr = []
    for point in resultats_scan:
        p = point["resultats"]
        mu_min = 1
        f_min = None
        for f in fichiers:
            if p[f][1] < mu_min:
                mu_min = p[f][1]
                f_min = f
        f_mu_min_arr.append(f_min)
        excluded_arr.append(f_min is not None)
    
    f_mu_min_arr = np.array(f_mu_min_arr, dtype=object)
    excluded_arr = np.array(excluded_arr)
    
    fig_6, axs_6 = plt.subplots(figsize=(15, 13))
    for i, f in enumerate(fichiers):
        mask = (f_mu_min_arr == f) & excluded_arr
        axs_6.scatter(my_arr[mask], mx_arr[mask], s=10, color=c[i], label=f)
    axs_6.scatter(my_arr[~excluded_arr], mx_arr[~excluded_arr], s=10, color="grey")

    # Region non-physique
    axs_6.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
    axs_6.fill_between(diag, diag, diag.max(), edgecolor='k',alpha=0.75, facecolor="k")
    
    axs_6.set_title("Analyse la plus contraignante")
    axs_6.set_ylabel(r"$M_X$ [GeV]")
    axs_6.set_xlabel(r"$M_Y$ [GeV]")
    axs_6.legend(fontsize=8)
    axs_6.grid(alpha=0.3)
    axs_6.plot([], [], color="k", linewidth=3, label="Région exclue")      
    

    # Remplissage combiné
    mu_min_subset_interp = np.nanmin(np.stack(mu_interp_subset_list), axis=0)
    axs_5.contourf(tab_my, tab_mx, mu_min_subset_interp, levels=[0, 1.0], colors="k", alpha=0.1)

    mu_min_all_interp = np.nanmin(np.stack(mu_interp_all_list), axis=0)
    axs_6.contourf(tab_my, tab_mx, mu_min_all_interp, levels=[0, 1.0], colors="k", alpha=0.1)
                       
    
    fig_glob.savefig(res_dir / "lims_exclusion.png", dpi=300, bbox_inches="tight")
    fig_1.savefig(res_dir / "mu_obs_par_analyse.png", dpi=300, bbox_inches="tight")
    fig_2.savefig(res_dir / "exclusion_par_analyse.png", dpi=300, bbox_inches="tight")
    fig_3.savefig(res_dir / "mu_obs_interp_par_analyse.png", dpi=300, bbox_inches="tight") 
    fig_4.savefig(res_dir / "exclusion_interp_par_analyse.png", dpi=300, bbox_inches="tight") 
    fig_5.savefig(res_dir / "atlas_susy_2018_22 complementaire.png", dpi=300, bbox_inches="tight") 
    fig_6.savefig(res_dir / "analyse_le_plus_contraignante.png", dpi=300, bbox_inches="tight") 
    plt.show()

    
    
resultats_scan = main(lam=LAM)
plot_excl(resultats_scan, lam = LAM)