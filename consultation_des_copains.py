# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:30:21 2026

@author: laeti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:45:07 2026

@author: aless
"""


import spey
import pandas as pd
import numpy as np
from numpy import inf

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt 

import re
from pathlib import Path


# Analyses a faire
analyses = ["atlas_susy_2018_22", "atlas_susy_2018_17", "atlas_exot_2018_06", "cms_sus_19_006", "cms_exo_20_004"]
fichiers = ["atlas_susy_2018_22_C","atlas_susy_2018_22_GGd", "atlas_susy_2018_22_SSd", "atlas_susy_2018_17", "atlas_exot_2018_06", "cms_sus_19_006", "cms_exo_20_004"]

base_dir = Path("C:/Users/laeti/Desktop/fac/stage/download")

# Extraction des points
def extract_points():
     
    points = []
     
    for run_dir in base_dir.glob("mass*"):
        name = run_dir.name
     
        my = float(re.search(r"mass5920005_([\d.]+)", name).group(1))
        mx = float(re.search(r"mass51_([\d.]+)", name).group(1))
     
        banner_candidates = list(run_dir.glob("**/*tag_1_banner.txt"))
        if not banner_candidates:
            print(f"Pas de banner trouvé pour {name}")
            continue
        banner_path = banner_candidates[0]

        banner_text = banner_path.read_text()
        xs = float(re.search(r"Integrated weight \(pb\)\s*:\s*([\d.eE+-]+)", banner_text).group(1))
        
        points.append({
            "xsection": xs,
            "mx": int(mx),
            "my": int(my),
        })
        
    return points


def extract(fichier):
    """ Extrait la luminosité et les covariance des analyses ayant cette info sur le fichier .info"""
    tree = ET.parse(fichier)
    root = tree.getroot()
    
    lumi = float(root.find("lumi").text)
    
    rows = []
    for region in root.findall(".//region"):
        region_id = region.attrib["id"]
        nobs = float(region.find("nobs").text)
        nb = float(region.find("nb").text)
        deltanb = float(region.find("deltanb").text)
        
        rows.append({
            "id": region_id,
            "nobs": nobs,
            "nb": nb,
            "deltanb": deltanb,
        })

    df = pd.DataFrame(rows)
    return df, lumi

def cov_mat(ids, covariances):
    """ Construit la matrice de covariance """
    matrix = np.zeros((len(ids), len(ids)))

    for i, l1 in enumerate(ids):
        for j, l2 in enumerate(ids):
            matrix[i, j] = covariances[l1][l2]
    return matrix


fichiers_cov = ["atlas_susy_2018_22", "atlas_susy_2018_17", "atlas_exot_2018_06", "cms_sus_19_006", "cms_exo_20_004"]
    


# extraction des fichiers atlas/cms de la covariance
def extract_cov(fichier):
    tree = ET.parse(fichier)
    root = tree.getroot()
    
    
    rows = []
    covariances = {}
    for region in root.findall(".//region"):
        region_id = region.attrib["id"]
        nobs = float(region.find("nobs").text)
        nb = float(region.find("nb").text)
        deltanb = float(region.find("deltanb").text)
        
        rows.append({
            "id": region_id,
            "nobs": nobs,
            "nb": nb,
            "deltanb": deltanb,
        })

        cov_tags = region.findall("covariance")
        if cov_tags:  # solo crea la entrada si hay al menos un tag <covariance>
            covariances[region_id] = {}
            for cov in cov_tags:
                cov_region = cov.attrib["region"]
                covariances[region_id][cov_region] = float(cov.text)

    df = pd.DataFrame(rows)
    return df,covariances


def matrices_cov():
    """Défini tous le matrices de covariances a utiliser """
    
    # liste de matrices : 
    matrix=[1,1,1,1,1]
    
    
    for j, f in enumerate(fichiers_cov):
        df_regions, covariances = extract_cov(f + ".info")
        df = df_regions
            
        if "cms" in f:
            ids = df["id"].tolist()
            if f == "cms_sus_19_006":
                ids = ids[12:]
                matrix[3] = cov_mat(ids,covariances)
                
            else:
                matrix[4] = cov_mat(ids,covariances)
        elif "atlas_susy_2018_22" in f : 
            #combine atlas
            matrix[0] =  np.array([
        [509.4952, 519.5835, 167.1863, 257.8663, 80.1488, 126.1517, 500.5227, 469.8060, 184.1580, 241.6788, 80.4041, 112.8283, 281.7265, 170.0266, 182.6437, 163.0106, 94.6281, 111.7590],
        [519.5835, 575.9265, 185.3997, 274.1625, 85.2139, 134.1240, 532.1540, 520.7701, 195.7962, 256.9520, 89.0906, 125.0769, 299.5305, 180.7716, 194.1861, 173.3123, 100.6082, 118.8218],
        [167.1863, 185.3997, 62.2498, 88.2572, 27.4317, 43.1766, 171.2311, 167.5681, 62.9715, 82.6404, 28.6667, 40.2270, 96.3799, 58.1394, 62.4832, 55.7667, 32.3727, 38.2152],
        [257.8663, 274.1625, 88.2572, 136.0062, 42.2728, 66.5361, 264.1049, 248.0093, 97.2165, 127.5816, 42.4451, 59.5899, 148.6553, 89.7565, 96.3734, 86.0139, 49.9313, 58.9973],
        [80.1488, 85.2139, 27.4317, 42.2728, 13.6808, 21.5331, 82.1207, 77.1159, 30.2285, 39.6702, 13.7482, 19.3103, 46.2228, 27.9089, 29.9663, 26.7451, 16.1660, 19.1096],
        [126.1517, 134.1240, 43.1766, 66.5361, 21.5331, 33.8925, 129.2554, 121.3781, 47.5787, 62.4396, 21.6393, 30.3938, 72.7533, 43.9277, 47.1661, 42.0960, 25.4449, 30.0779],
        [500.5227, 532.1540, 171.2311, 264.1049, 82.1207, 129.2554, 556.4829, 522.7954, 204.9292, 268.9377, 89.4371, 125.6202, 300.6955, 181.5569, 194.9413, 173.9863, 100.9995, 119.3379],
        [469.8060, 520.7701, 167.5681, 248.0093, 77.1159, 121.3781, 522.7954, 511.8249, 192.4400, 252.5476, 87.5238, 122.9330, 282.3698, 170.4149, 183.0608, 163.3829, 94.8442, 112.0142],
        [184.1580, 195.7962, 62.9715, 97.2165, 30.2285, 47.5787, 204.9292, 192.4400, 78.6440, 103.2080, 32.9216, 46.2196, 110.6854, 66.8005, 71.7575, 64.0440, 37.1777, 43.9082],
        [241.6788, 256.9520, 82.6404, 127.5816, 39.6702, 62.4396, 268.9377, 252.5476, 103.2080, 135.4445, 43.2045, 60.6561, 145.2574, 87.6652, 94.1706, 84.0478, 48.7900, 57.6226],
        [80.4041, 89.0906, 28.6667, 42.4451, 13.7482, 21.6393, 89.4371, 87.5238, 32.9216, 43.2045, 15.5970, 21.9165, 48.3257, 29.1653, 31.3296, 27.9619, 16.9088, 19.9790],
        [112.8283, 125.0769, 40.2270, 59.5899, 19.3103, 30.3938, 125.6202, 122.9330, 46.2196, 60.6561, 21.9165, 32.0931, 67.8458, 42.6729, 43.9846, 39.2565, 23.7496, 29.2320],
        [281.7265, 299.5305, 96.3799, 148.6553, 46.2228, 72.7533, 300.6955, 282.3698, 110.6854, 145.2574, 48.3257, 67.8458, 176.3029, 106.4961, 114.2974, 102.0112, 59.1940, 69.9722],
        [170.0266, 180.7716, 58.1394, 89.7565, 27.9089, 43.9277, 181.5569, 170.4149, 66.8005, 87.6652, 29.1653, 42.6729, 106.4961, 67.0375, 69.0416, 61.6201, 35.7563, 44.0280],
        [182.6437, 194.1861, 62.4832, 96.3734, 29.9663, 47.1661, 194.9413, 183.0608, 71.7575, 94.1706, 31.3296, 43.9846, 114.2974, 69.0416, 74.0992, 66.1340, 38.3756, 45.3631],
        [163.0106, 173.3123, 55.7667, 86.0139, 26.7451, 42.0960, 173.9863, 163.3829, 64.0440, 84.0478, 27.9619, 39.2565, 102.0112, 61.6201, 66.1340, 59.0250, 34.2505, 40.4869],
        [94.6281, 100.6082, 32.3727, 49.9313, 16.1660, 25.4449, 100.9995, 94.8442, 37.1777, 48.7900, 16.9088, 23.7496, 59.1940, 35.7563, 38.3756, 34.2505, 20.6940, 24.4722],
        [111.7590, 118.8218, 38.2152, 58.9973, 19.1096, 30.0779, 119.3379, 112.0142, 43.9082, 57.6226, 19.9790, 29.2320, 69.9722, 44.0280, 45.3631, 40.4869, 24.4722, 30.1335],
    ]
    )        
            #production gluinos (Ggd)
            matrix[1] = np.array([
        [8481.8693, 5913.1358, 2706.0344, 2828.9717, 2083.8837, 2041.0406, 588.5953, 783.3942, 1006.2091, 229.5429, 348.8736, 489.7939, 105.9025, 82.9911, 108.8065, 44.6844, 31.8119, 51.3536],
        [5913.1358, 4488.6003, 1886.5121, 1972.2178, 1451.4357, 1422.9116, 410.3392, 545.6381, 730.9833, 160.0259, 242.9923, 355.8218, 73.7616, 57.7479, 75.7843, 30.9205, 22.1134, 35.6597],
        [2706.0344, 1886.5121, 899.6731, 940.5460, 693.1280, 678.5832, 195.6898, 260.5675, 334.8304, 76.3159, 116.0401, 162.9859, 35.2246, 27.6165, 36.1905, 14.9079, 10.5909, 17.1052],
        [2828.9717, 1972.2178, 940.5460, 983.2757, 724.6173, 709.4117, 204.5801, 272.4053, 350.0420, 79.7830, 121.3119, 170.3905, 36.8249, 28.8711, 37.8347, 15.5852, 11.0720, 17.8823],
        [2083.8837, 1451.4357, 693.1280, 724.6173, 556.4829, 522.7954, 150.7637, 209.1986, 268.8210, 58.7955, 93.1636, 130.8544, 28.2803, 22.1721, 29.0558, 11.9640, 8.1484, 13.7330],
        [2041.0406, 1422.9116, 678.5832, 709.4117, 522.7954, 511.8249, 147.6001, 196.5344, 252.5476, 57.5617, 87.5238, 122.9330, 26.5683, 20.8299, 27.2969, 11.2444, 7.9882, 12.9017],
        [588.5953, 410.3392, 195.6898, 204.5801, 150.7637, 147.6001, 42.5649, 56.6766, 72.8297, 16.5996, 25.2401, 35.4514, 7.6618, 6.0069, 7.8719, 3.2427, 2.3036, 3.7206],
        [783.3942, 545.6381, 260.5675, 272.4053, 209.1986, 196.5344, 56.6766, 78.6440, 101.0579, 22.1030, 35.0230, 49.1921, 10.6314, 8.3352, 10.9229, 4.4976, 3.0632, 5.1626],
        [1006.2091, 730.9833, 334.8304, 350.0420, 268.8210, 252.5476, 72.8297, 101.0579, 135.4445, 28.4024, 45.0047, 65.9305, 13.6614, 10.7059, 14.0360, 5.7619, 3.9307, 6.6246],
        [229.5429, 160.0259, 76.3159, 79.7830, 58.7955, 57.5617, 16.5996, 22.1030, 28.4024, 6.4736, 9.8433, 13.8255, 2.9880, 2.3426, 3.0699, 1.2646, 0.8984, 1.4510],
        [348.8736, 242.9923, 116.0401, 121.3119, 93.1636, 87.5238, 25.2401, 35.0230, 45.0047, 9.8433, 15.5970, 21.9070, 4.7346, 3.7119, 4.8644, 2.0029, 1.3642, 2.2991],
        [489.7939, 355.8218, 162.9859, 170.3905, 130.8544, 122.9330, 35.4514, 49.1921, 65.9305, 13.8255, 21.9070, 32.0931, 6.6500, 5.2113, 6.8323, 2.8047, 1.9133, 3.2247],
        [105.9025, 73.7616, 35.2246, 36.8249, 28.2803, 26.5683, 7.6618, 10.6314, 13.6614, 2.9880, 4.7346, 6.6500, 1.4372, 1.1268, 1.4766, 0.6080, 0.4141, 0.6979],
        [82.9911, 57.7479, 27.6165, 28.8711, 22.1721, 20.8299, 6.0069, 8.3352, 10.7059, 2.3426, 3.7119, 5.2113, 1.1268, 0.9214, 1.1577, 0.4752, 0.3242, 0.5464],
        [108.8065, 75.7843, 36.1905, 37.8347, 29.0558, 27.2969, 7.8719, 10.9229, 14.0360, 3.0699, 4.8644, 6.8323, 1.4766, 1.1577, 1.5171, 0.6247, 0.4255, 0.7170],
        [44.6844, 30.9205, 14.9079, 15.5852, 11.9640, 11.2444, 3.2427, 4.4976, 5.7619, 1.2646, 2.0029, 2.8047, 0.6080, 0.4752, 0.6247, 0.3501, 0.2078, 0.3355],
        [31.8119, 22.1134, 10.5909, 11.0720, 8.1484, 7.9882, 2.3036, 3.0632, 3.9307, 0.8984, 1.3642, 1.9133, 0.4141, 0.3242, 0.4255, 0.2078, 0.1473, 0.2282],
        [51.3536, 35.6597, 17.1052, 17.8823, 13.7330, 12.9017, 3.7206, 5.1626, 6.6246, 1.4510, 2.2991, 3.2247, 0.6979, 0.5464, 0.7170, 0.3355, 0.2282, 0.3856],
    ])   
            
            #
            matrix[2] = np.array([
        [6577.4526, 3830.9732, 2696.9128, 2272.0960, 1793.2633, 1827.1498, 600.7027, 887.9121, 275.9767, 443.2435, 138.0867, 260.1218, 81.9557, 141.4968, 6707.0454, 4768.2246, 2085.0883, 2293.7927, 1798.3848, 1688.0193, 647.2979, 868.3551, 282.6127, 405.3937],
        [3830.9732, 2425.2524, 1637.6327, 1379.6734, 1088.9142, 1109.4909, 364.7617, 539.1624, 167.5799, 269.1485, 80.3132, 157.9525, 49.7656, 85.9204, 3900.5218, 2771.4894, 1211.2525, 1392.8482, 1092.0241, 1025.0074, 393.0554, 527.2869, 171.6095, 246.1652],
        [2696.9128, 1637.6327, 1201.9116, 971.2568, 766.5694, 814.6600, 267.8316, 379.5573, 117.9722, 189.4741, 56.5385, 111.1947, 35.0338, 60.4859, 2745.8734, 1951.0618, 891.3163, 1021.8844, 768.7587, 752.3133, 288.6229, 371.1973, 120.8090, 173.2944],
        [2272.0960, 1379.6734, 971.2568, 851.6343, 645.5393, 658.3214, 216.4330, 319.9144, 103.6224, 159.7005, 47.7003, 93.6794, 30.7451, 53.2291, 2318.7550, 1648.4672, 720.8552, 861.2620, 647.9232, 608.4361, 233.4248, 312.9932, 106.2123, 152.3562],
        [1793.2633, 1088.9142, 766.5694, 645.5393, 509.4952, 519.5835, 170.8208, 252.4941, 78.4790, 126.0444, 37.6477, 73.9369, 23.2951, 40.2761, 1829.3572, 1300.5408, 568.7111, 652.2820, 511.1722, 480.0192, 184.1580, 246.9327, 80.4041, 115.3356],
        [1827.1498, 1109.4909, 814.6600, 658.3214, 519.5835, 575.9265, 189.3444, 268.2128, 79.8880, 133.8912, 39.9527, 75.3341, 23.7353, 40.9182, 1856.5584, 1378.1364, 629.8454, 692.3234, 520.8315, 509.6899, 203.9543, 262.1959, 85.2995, 122.3579],
        [600.7027, 364.7617, 267.8316, 216.4330, 170.8208, 189.3444, 62.2498, 88.1790, 26.2644, 44.0187, 13.1351, 24.7672, 7.8033, 13.4525, 610.3712, 453.0829, 207.0711, 227.6116, 171.2311, 167.5681, 67.0530, 86.2008, 28.0435, 40.2270],
        [887.9121, 539.1624, 379.5573, 319.9144, 252.4941, 268.2128, 88.1790, 136.0062, 38.8220, 65.0650, 19.4152, 36.6089, 11.5343, 19.8844, 902.2033, 669.7119, 292.6911, 322.6706, 253.1006, 237.4557, 94.9792, 127.4154, 41.4517, 59.4604],
        [275.9767, 167.5799, 117.9722, 103.6224, 78.4790, 79.8880, 26.2644, 38.8220, 13.6808, 20.2148, 6.2977, 11.8632, 3.8951, 6.7345, 280.5309, 199.2163, 87.0138, 104.6119, 78.6990, 73.8344, 28.2990, 37.9821, 13.4389, 19.2774],
        [443.2435, 269.1485, 189.4741, 159.7005, 126.0444, 133.8912, 44.0187, 65.0650, 20.2148, 33.8925, 10.1232, 19.0614, 6.0031, 10.3644, 450.3777, 334.3186, 146.1107, 161.0764, 126.3472, 118.5373, 47.4134, 63.6054, 21.5841, 30.9613],
        [138.0867, 80.3132, 56.5385, 47.7003, 37.6477, 39.9527, 13.1351, 19.4152, 6.2977, 10.1232, 3.2984, 5.6934, 1.7930, 3.0863, 139.9540, 103.9321, 45.3955, 48.0162, 37.7016, 35.3355, 14.1337, 18.9797, 6.4406, 9.2388],
        [260.1218, 157.9525, 111.1947, 93.6794, 73.9369, 75.3341, 24.7672, 36.6089, 11.8632, 19.0614, 5.6934, 11.1813, 3.5214, 6.0883, 264.8442, 188.1832, 82.2436, 94.5739, 74.1481, 69.5977, 26.6883, 35.8026, 12.1494, 17.4277],
        [81.9557, 49.7656, 35.0338, 30.7451, 23.2951, 23.7353, 7.8033, 11.5343, 3.8951, 6.0031, 1.7930, 3.5214, 1.1557, 2.0009, 83.4770, 59.3140, 25.9226, 31.0662, 23.3709, 21.9367, 8.4120, 11.2847, 3.9909, 5.7247],
        [141.4968, 85.9204, 60.4859, 53.2291, 40.2761, 40.9182, 13.4525, 19.8844, 6.7345, 10.3644, 3.0863, 6.0883, 2.0009, 3.9307, 143.2091, 101.5745, 44.3089, 53.6358, 40.3500, 37.8176, 14.4793, 19.4542, 6.8903, 9.8838],
        [6707.0454, 3900.5218, 2745.8734, 2318.7550, 1829.3572, 1856.5584, 610.3712, 902.2033, 280.5309, 450.3777, 139.9540, 264.8442, 83.4770, 143.2091, 8481.8693, 6033.1124, 2639.6363, 2541.3596, 1995.2078, 1870.9539, 716.7224, 962.4608, 312.7986, 448.6938],
        [4768.2246, 2771.4894, 1951.0618, 1648.4672, 1300.5408, 1378.1364, 453.0829, 669.7119, 199.2163, 334.3186, 103.9321, 188.1832, 59.3140, 101.5745, 6033.1124, 4488.6003, 1963.8740, 1806.7230, 1419.1816, 1330.1130, 532.2498, 714.7392, 232.1926, 333.0686],
        [2085.0883, 1211.2525, 891.3163, 720.8552, 568.7111, 629.8454, 207.0711, 292.6911, 87.0138, 146.1107, 45.3955, 82.2436, 25.9226, 44.3089, 2639.6363, 1963.8740, 899.6731, 825.3771, 620.9272, 607.8974, 243.3581, 312.7161, 101.5351, 145.6470],
        [2293.7927, 1392.8482, 1021.8844, 861.2620, 652.2820, 692.3234, 227.6116, 322.6706, 104.6119, 161.0764, 48.0162, 94.5739, 31.0662, 53.6358, 2541.3596, 1806.7230, 825.3771, 983.2757, 710.1250, 694.9339, 266.7302, 343.0411, 116.4089, 166.9826],
        [1798.3848, 1092.0241, 768.7587, 647.9232, 511.1722, 520.8315, 171.2311, 253.1006, 78.6990, 126.3472, 37.7016, 74.1481, 23.3709, 40.3500, 1995.2078, 1419.1816, 620.9272, 710.1250, 556.4829, 522.7954, 200.6599, 268.9377, 87.5738, 125.6202],
        [1688.0193, 1025.0074, 752.3133, 608.4361, 480.0192, 509.6899, 167.5681, 237.4557, 73.8344, 118.5373, 35.3355, 69.5977, 21.9367, 37.8176, 1870.9539, 1330.1130, 607.8974, 694.9339, 522.7954, 511.8249, 196.4491, 252.5476, 82.1995, 117.9110],
        [647.2979, 393.0554, 288.6229, 233.4248, 184.1580, 203.9543, 67.0530, 94.9792, 28.2990, 47.4134, 14.1337, 26.6883, 8.4120, 14.4793, 716.7224, 532.2498, 243.3581, 266.7302, 200.6599, 196.4491, 78.6440, 101.0579, 32.8787, 47.1629],
        [868.3551, 527.2869, 371.1973, 312.9932, 246.9327, 262.1959, 86.2008, 127.4154, 37.9821, 63.6054, 18.9797, 35.8026, 11.2847, 19.4542, 962.4608, 714.7392, 312.7161, 343.0411, 268.9377, 252.5476, 101.0579, 135.4445, 44.0862, 63.2395],
        [282.6127, 171.6095, 120.8090, 106.2123, 80.4041, 85.2995, 28.0435, 41.4517, 13.4389, 21.5841, 6.4406, 12.1494, 3.9909, 6.8903, 312.7986, 232.1926, 101.5351, 116.4089, 87.5738, 82.1995, 32.8787, 44.0862, 15.5970, 22.3731],
        [405.3937, 246.1652, 173.2944, 152.3562, 115.3356, 122.3579, 40.2270, 59.4604, 19.2774, 30.9613, 9.2388, 17.4277, 5.7247, 9.8838, 448.6938, 333.0686, 145.6470, 166.9826, 125.6202, 117.9110, 47.1629, 63.2395, 22.3731, 32.0931],
    ])
    
    return matrix

 
def ext_eff(analyses, point_dossier):
    """Extrait nf, n0 et calcule l'efficacité par region."""

    rows = []
    
    inner_candidates = list((Path(base_dir) / point_dossier).glob("mass*"))
    if not inner_candidates:
        print(f"Pas de sous-dossier 'mass*' trouvé dans {point_dossier}")
        return pd.DataFrame(rows, columns=["id", "efficiency"])
    inner_dir = inner_candidates[0]
 
    for analysis in analyses:
        cutflow_dir = inner_dir / "Output"/ "SAF"/ "defaultset"/analysis/ "Cutflows"
 
        for saf_file in cutflow_dir.glob("*.saf"):
            region = saf_file.stem
            text = saf_file.read_text()
 
            # n0: InitialCounter
            init_block = re.search(r"<InitialCounter>(.*?)</InitialCounter>", text, re.DOTALL).group(1)
            n0_line = re.search(r"^.*# nentries", init_block, re.MULTILINE).group(0)
            n0 = float(n0_line.split()[0])
 
            # nf: <Counter>
            last_counter = re.findall(r"<Counter>(.*?)</Counter>", text, re.DOTALL)[-1]
            nf_line = re.search(r"^.*# nentries", last_counter, re.MULTILINE).group(0)
            nf = float(nf_line.split()[0])
 
            efficiency = nf / n0
 
            rows.append({"id": region,"efficiency": efficiency})
 
    return pd.DataFrame(rows, columns=["id", "efficiency"])
 
    
def main():
    points = extract_points()
    matrix = matrices_cov()
    
    resultats_scan = []
    for point in points :
        print(point)
        # Pour la simulation
        xsection = point["xsection"]# 1.508068e-01
        mx = point["mx"] # masse de la particule de matière noir
        my = point["my"] # masse du médiateur
        pattern = f"mass5920005_{float(point['my'])}_mass51_{float(point['mx'])}_dmf3d33_1.0_xs_*"
        print(pattern)
        matches = list(Path(base_dir).glob(pattern))
        point_dossier = matches[0].name
        


        df_eff = ext_eff(analyses,point_dossier) # extraire les efficacitées
        print(df_eff)
        
        
        resultats_par_analyse = {}
        for j, f in enumerate(fichiers):
            if f in ("atlas_susy_2018_22_C", "atlas_susy_2018_22_GGd", "atlas_susy_2018_22_SSd"):
                # Charger toutes les régions du fichier .info commun
                df_regions, lumi = extract("atlas_susy_2018_22.info")
                # Filtrer uniquement les régions de la sous-analyse courante
                prefix_map = {
                    "atlas_susy_2018_22_C":   "MB_C_",
                    "atlas_susy_2018_22_GGd": "MB_GGd_",
                    "atlas_susy_2018_22_SSd": "MB_SSd_",
                }
                prefix = prefix_map[f]
                df_regions = df_regions[df_regions["id"].str.startswith(prefix)].reset_index(drop=True)
                df = df_regions.merge(df_eff, on="id", how="left")
                df["efficiency"] = df["efficiency"].fillna(0.0)

            else:
                df_regions, lumi = extract(f + ".info")
                df = df_regions.merge(df_eff, on="id", how="left")
                df["efficiency"] = df["efficiency"].fillna(0.0)

            print("\n", f)
            nsignal = xsection * lumi * 1000 * df["efficiency"].to_numpy()
            
            if "cms" in f:
                
                if f == "cms_sus_19_006":
                
                    mat_cov = matrix[3]
    
                    stat_model = spey.get_backend("default.correlated_background")(
                        signal_yields=nsignal.tolist()[12:],
                        background_yields=df["nb"].tolist()[12:],
                        data=df["nobs"].tolist()[12:],
                        covariance_matrix=mat_cov.tolist(),
                        analysis=f,
                    )
                else :
                    mat_cov = matrix[4]
                    stat_model = spey.get_backend("default.correlated_background")(
                        signal_yields=nsignal.tolist(),
                        background_yields=df["nb"].tolist(),
                        data=df["nobs"].tolist(),
                        covariance_matrix=mat_cov.tolist(),
                        analysis=f,
                    )
                
            elif "atlas_susy_2018_22_C" in f:

                mat_cov = matrix[0]
                stat_model = spey.get_backend("default.correlated_background")(
                    signal_yields=nsignal.tolist(),
                    background_yields=df["nb"].tolist(),
                    data=df["nobs"].tolist(),
                    covariance_matrix=mat_cov.tolist(),
                    analysis=f,
                )



            elif "atlas_susy_2018_22_GGd" in f:

                mat_cov = matrix[1]
                stat_model = spey.get_backend("default.correlated_background")(
                    signal_yields=nsignal.tolist(),
                    background_yields=df["nb"].tolist(),
                    data=df["nobs"].tolist(),
                    covariance_matrix=mat_cov.tolist(),
                    analysis=f,
                )


            elif "atlas_susy_2018_22_SSd" in f:

                mat_cov = matrix[2]
                stat_model = spey.get_backend("default.correlated_background")(
                    signal_yields=nsignal.tolist(),
                    background_yields=df["nb"].tolist(),
                    data=df["nobs"].tolist(),
                    covariance_matrix=mat_cov.tolist(),
                    analysis=f,
                )
                
            """
                        elif "atlas_exot_2018_06" in f : # raj noms exacts des dossiers sans json
                        
                            i_best = df[df["best"] == 1].iloc[0].name
                            
                            stat_model = spey.get_backend("default.uncorrelated_background")(
                                signal_yields=[nsignal[i_best]],
                                background_yields=[df["nb"].iloc[i_best]],
                                data=[df["nobs"].iloc[i_best]],
                                absolute_uncertainties=[df["deltanb"].iloc[i_best]],
                                analysis=df["id"].iloc[i_best],
                            )
                        
                        elif "atlas_susy_2018_17" in f :
                            
                            i_best = df[df["best"] == 1].iloc[0].name
                            
                            stat_model = spey.get_backend("default.uncorrelated_background")(
                                signal_yields=[nsignal[i_best]],
                                background_yields=[df["nb"].iloc[i_best]],
                                data=[df["nobs"].iloc[i_best]],
                                absolute_uncertainties=[df["deltanb"].iloc[i_best]],
                                analysis=df["id"].iloc[i_best],
                            )
            """
            
            try:
                mu_obs = stat_model.poi_upper_limit(expected=spey.ExpectationType.observed)
                one_minus_cls_obs = stat_model.exclusion_confidence_level(expected=spey.ExpectationType.observed)[0]
                rejetee = one_minus_cls_obs >= 0.95
            except ValueError as e:
                 print(f"problème")
                 mu_obs= np.nan
                 rejetee = False
                 
                 
            resultats_par_analyse[f] = [rejetee, mu_obs]                
        resultats_scan.append({"mx": mx, "my": my, "resultats": resultats_par_analyse})
    return resultats_scan


def plot_excl(resultats_scan) :
    
    
    mx_arr = np.array([point["mx"] for point in resultats_scan])
    my_arr = np.array([point["my"] for point in resultats_scan])

    # Tracé des limites d'exclusion

    lam = 1
    c = ["r", "g", "b", "orange", "m","pink","k","yellow"]

    fig, axs = plt.subplots(3, 3, figsize=(15, 13))
    for i, (ax, f) in enumerate(zip(axs.flat, fichiers)):
        mu_array = np.array([point["resultats"][f][1] for point in resultats_scan])
        print(mu_array[mu_array != inf])
        mu_array[mu_array == inf] = 1000
        if i == len(fichiers)-1 : 
            ax = axs[2,1]
        
        sc = ax.scatter(my_arr, mx_arr, c=mu_array, cmap="inferno")
        fig.colorbar(sc, ax=ax, label=r"$\mu_{obs}$")
    
        ax.set_title(f)
        ax.set_ylabel("MX [GeV]")
        ax.set_xlabel("MY [GeV]")
        ax.grid(alpha=0.3)
    
    axs[2, 0].axis("off")
    axs[2, 2].axis("off")
    fig.suptitle(r"Signal strength $\mu_{obs}$ par analyse")
    plt.tight_layout()
    plt.show()

    
    # Subplots
    diag = np.linspace(min(mx_arr.min(), my_arr.min()), max(my_arr.max(), mx_arr.max()))

    fig, axs = plt.subplots(3, 3, figsize=(15, 13))
    for i, (ax, f) in enumerate(zip(axs.flat, fichiers)):
        if i == len(fichiers) - 1:
            ax = axs[2, 1]

        rejetee_arr = np.array([point["resultats"][f][0] for point in resultats_scan])
        ax.plot(diag, diag, "--", color="gray", label=r"$M_X = M_Y$")
        ax.scatter(my_arr[rejetee_arr], mx_arr[rejetee_arr], color=c[i], label="Exclu")

        ax.set_title(f)
        ax.set_ylabel(r"$M_X$ [GeV]")
        ax.set_xlabel(r"$M_Y$ [GeV]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.5)

    axs[2, 0].axis("off")
    axs[2, 2].axis("off")
    fig.suptitle(rf"Exclusion des paramètres d'après le LHC ($\lambda$ = {lam})")
    plt.tight_layout()
    plt.show()

    
    """ 
    # Dans un même plot
    plt.figure(figsize=(12, 6))
    diag = np.linspace(min(mx_arr.min(), my_arr.min()), max(my_arr.max(), mx_arr.max()))
    plt.plot(diag)
    for i, f in enumerate(fichiers):
        rejetee_arr = np.array([point["resultats"][f][0] for point in resultats_scan])
        plt.scatter(my_arr[rejetee_arr], mx_arr[rejetee_arr], color=c[i], label=f)
 
    plt.title(rf"Exclusion des paramètres d'après le LHC ($\lambda$ = {lam})")
    plt.ylabel(r"$M_X$ [GeV]")
    plt.xlabel(r"$M_Y$ [GeV]")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
    """
    



resultats_scan = main()
plot_excl(resultats_scan)
