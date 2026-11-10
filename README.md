 # Test_DM_F3Sbr

## Projet réalisé par : 
- Victor Alessandro Sales Lopez
- Sidi-Mansour Malagouen
- Laëtitia Renard

  ## Présentation du projet :
 Notre projet de stage consistait à imposer des contraintes sur un modèle de matière noire particulier : F3Sbr. 
 On suppose dans notre cas que la matière noire est une particule, qui interagit avec le modèle standard grâce à une particule médiatrice. 
 Le __rapport.pdf__ présente le projet plus en détail, la physique associée, le contexte et les résultats obtenus à la fin du stage. 

 Les objectifs sont : 
 
 - Prise en main des outils madgraph5 et Madanalysis5
 - Prise en main du terminal Linux
 - Compréhension via de premières simulations de la physique et des enjeux associés au projet
 - Création d'un code pour réinterpréter les données expérimentales obtenues au LHC ces dernières années avec nos simulations, sur Madanalysis5, pour en déduire des contraintes sur le modèle F3Sbr
 - Rédaction d'un rapport de stage complet et rigoureux 
 


## Installation :
Dans le fichier __journal_de_bord.pdf__, on trouve le déroulé semaine par semaine du projet , ainsi que les détails techniques,  à savoir : les différents modules/ packages à installer, les erreurs rencontrées et leurs solutions, l'emplacement des données dans Madanalysis5 et autres. 


## Description des codes

Les fichiers `.py` dans le dossier `Code` permettent de réinterpréter les résultats des analyses CMS/ATLAS présentes dans le dossier `Analyses`.

### `calcul_limites_exclusion.py`

C'est le code principal, qui calcule les limites d'exclusion pour le Run 2 et fait des projections pour le Run 3 et le HL-LHC.

Les principales fonctions sont :

- `extract_points()` : récupère les masses et les sections efficaces des simulations.
- `extract()`, `extract_cov()`, `cov_mat()` : récupèrent les informations des analyses et leurs matrices de covariance.
- `load_static_regions()`, `ext_eff()` : récupèrent les régions de signal et leurs efficacités.
- `load_pyhf_workspaces()`, `build_signal_patch()` : préparent les modèles statistiques pyhf.
- `best_region()`, `get_stat_result()` : calculent les limites d'exclusion avec Spey.
- `scale_static_regions()`, `scale_cov_matrices()`, `scale_pyhf_workspace()` : adaptent les données et leurs incertitudes aux luminosités du Run 3 et du HL-LHC.
- `main()` : lance les calculs et sauvegarde les résultats.
- `interp_mu()`, `plot_excl()` : interpolent les résultats et tracent les graphiques d'exclusion.

### `comparaison_runs_lhc.py`

Ce code utilise les résultats du premier pour comparer les exclusions obtenues avec le Run 2, le Run 3 et le HL-LHC.

- `import_res()` : charge les résultats des trois runs.
- `grille_commune()` : crée une grille commune pour comparer les résultats.
- `zone_non_physique()` : affiche la zone non physique.
- `mu_min_global()` : récupère la limite la plus contraignante parmi les analyses.
- `plot_comp()` : trace les comparaisons entre les trois runs, pour chaque analyse et globalement.

Les résultats des calculs sont sauvegardés dans des fichiers `.pkl`, et les graphiques dans des fichiers `.png`.



## Utilisation

Pour utiliser les codes, il suffit de les ouvrir dans un environnement Python comme **Spyder** et de les exécuter.

### Organisation des dossiers

Par defaut, les fichiers sont supposés être organisés de la manière suivante :

```text
stage/
├── calcul_limites_exclusion.py
├── comparaison_runs_lhc.py
├── Points/
│   ├── mass5920005_.../
│   ├── mass5920005_.../
│   └── ...
├── atlas_susy_2018_22.info
├── atlas_susy_2018_22_C.json
├── atlas_susy_2018_22_GGd.json
├── atlas_susy_2018_22_SSd.json
├── ... (autres fichiers .info)
└── Resultats/
    ├── Run_2/
    ├── Run_3/
    ├── HL-LHC/
    └── Comparaison/
```

Les chemins d'accès aux dossiers sont définis à la fin des codes et correspondent à l'organisation présentée ci-dessus :

- `base_dir` : chemin vers le dossier `stage/Points/`.
- `info_dir` : chemin vers le dossier `stage/`, contenant les fichiers `.info` et `.json`.
- `res_dir` : chemin vers le dossier `stage/Resultats/`, où sont sauvegardés les résultats et les graphiques.

Si les fichiers sont organisés différemment sur votre ordinateur, il suffit de modifier ces chemins dans les codes ou dans l'ordinateur.

### `calcul_limites_exclusion.py`

Ouvrir le fichier dans Spyder et l'exécuter. Dans la console, le programme demande :
- La valeur de `Lambda`. (Il faut bien vérifier que le dossier `Points` contient des points simulés pour la valeur de Lambda choisie, sinon le code ne pourra pas effectuer les calculs.)
- Le run à analyser (`2`, `3` ou `HL`).

Les calculs se lancent ensuite automatiquement et les résultats sont sauvegardés dans le dossier `Resultats`.

Les luminosités et le facteur d'amélioration des incertitudes peuvent aussi être modifiés directement dans le code.

### `comparaison_runs_lhc.py`

Une fois les calculs effectués pour les trois runs, ouvrir ce fichier dans Spyder et l'exécuter.

Il suffit d'entrer la valeur de `Lambda` dans la console pour afficher les comparaisons entre les trois runs.

Les graphiques sont sauvegardés dans `Resultats/Comparaison/`.

