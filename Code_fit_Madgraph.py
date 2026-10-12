 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:31:20 2026

@author: aless
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.optimize import curve_fit


#Éq. pour les ajustements
def quad(x,a,b,c):
    return a*x**2 + b*x +c

def poly4(x,a,b,c,d,e):
    return a*x**4 + b*x**3 +c*x**2 + d*x + e

def partial_widths(lamF3d3x3, MYF3d3, MXs):
    return (lamF3d3x3**2*MXs**4)/(32*cmath.pi*abs(MYF3d3)**3)+(-(lamF3d3x3**2*MXs**2*MYF3d3**2))/(16*cmath.pi*abs(MYF3d3)**3)+(lamF3d3x3**2*MYF3d3**4)/(32*cmath.pi*abs(MYF3d3)**3)

### ÉQ. Analytique des PARTIAL WIDTHS : valeurs fixées
lamF3d3x3 = 0.7
MYF3d3 = 1000  # masse du médiateur en GeV
MXs = 10       # masse de la matière noire en GeV

# En faisant varier le couplage et "Auto" pour la largeyr du mediateur
lam = np.arange(0.5, 8.6, 1) # couplage
larg = np.array([2.4862986514816625,22.37668786333496,62.15746628704156,121.82863392260145,201.39019077001464,300.8421368292811,420.1844721004009,559.417196583374,718.5403102782004]) # largeur de du mediateur
cs = np.array([0.02906, 0.03575, 0.05339, 0.09839, 0.2566, 0.7484, 2.105, 5.327, 11.47]) # section efficace de la simulation
cs_err = np.array([3.9e-05, 6.5e-05, 1.2e-04, 2.8e-04, 5.8e-04, 2.0e-03, 5.8e-03, 1.1e-02, 3.0e-02]) # erreur sur la section efficace

# Ajustement de la largeur en fonction de lambda
params_larg, pcov_larg = curve_fit(quad, lam, larg)
a, b, c = params_larg
a_err, b_err, c_err = np.sqrt(np.diag(pcov_larg))
# Ajustement de la section efficace en fonction de lambda
params_cs, pcov_cs = curve_fit(poly4,lam, cs, sigma = cs_err, absolute_sigma = True)
l, m, n, o, p = params_cs
l_err, m_err, n_err, o_err, p_err = np.sqrt(np.diag(pcov_cs))


lam_fit = np.linspace(lam.min()-0.5, lam.max()+0.5, 300)
larg_fit = quad(lam_fit, *params_larg)
cs_fit = poly4(lam_fit, *params_cs)


larg_theo_lam = partial_widths(lam, MYF3d3, MXs)

plt.figure()
plt.plot(lam, larg, "ok")
plt.xlabel(r"Couplage $\lambda$")
plt.plot(lam_fit, larg_fit, "-", label="Ajustement quadratique", color = "r")
plt.plot(lam, larg_theo_lam, "--", label="Formule analytique", color = "c")
plt.ylabel("Largeur du mediateur wyf3d3 [GeV]")
plt.title("Largeur du mediateur en fonction du couplage")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.errorbar(lam, cs, yerr=cs_err, fmt="o", capsize=3, label="Données", color = "k")
plt.xlabel(r"Couplage $\lambda$")
plt.ylabel(r"Section efficace $\sigma$ [pb]")
plt.plot(lam_fit, cs_fit, "-", label="Ajustement polynomial degré 4", color = "c")
plt.title("Section efficace en fonction du couplage")
plt.grid()
plt.legend()
plt.show()




# Avec couplage lamdba=0.7 et en faisant varier la masse de la particule --> mxs
mxs = np.arange(200, 831, 70)
cs_mxs = np.array([0.02918,0.02902,0.02879,0.02872,0.02855,0.02832,0.02832,0.02828,0.02809,0.0279])
cs_mxs_err = np.array([4.4e-5,3.6e-5,4.6e-5,5.3e-5,5.1e-5,4.7e-5,3.7e-5,4.5e-5,5e-5,5.1e-5])
larg_mxs = np.array([4.491989113825653,4.1893764896878025,3.812358179636822,3.3731719295358067,2.886864148232825,2.3712899075609175,1.8471129423380988,1.337805650367355,0.8696490924366467,0.47173299231890464])

larg_theo_mxs = partial_widths(lamF3d3x3, MYF3d3, mxs)


plt.figure()
plt.plot(mxs, larg_mxs, "co", label = "Données")
plt.plot(mxs, larg_theo_mxs, "--", label="Formule analytique", color = "lightblue")
plt.xlabel(r"Masse de la particule de matière noire $mxs$ [GeV]")
plt.ylabel("Largeur du mediateur wyf3d3 [GeV^-1]")
plt.title("Largeur du mediateur en fonction du couplage")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.errorbar(mxs, cs_mxs, yerr=cs_mxs_err, fmt="o", capsize=3, label="Données")
plt.xlabel(r"Masse de la particule de matière noire $mxs$ [GeV]")
plt.ylabel(r"Section efficace $\sigma$ [pb]")
plt.title("Section efficace en fonction du couplage")
plt.grid()
plt.legend()
plt.show()


# Avec couplage lamdba=0.7 et en faisant varier la masse de la particule --> myf3d3

myf3d3 = np.arange(200, 2001, 300)
cs_myf3d3 = np.array([1.875637e+02, 1.899790e+00, 1.258021e-01, 1.543741e-02, 2.485075e-03, 4.619699e-04, 9.433821e-05])
cs_myf3d3_err = np.array([2.526413e-01, 2.043147e-03, 1.994815e-04, 2.876338e-05, 2.995472e-06, 6.688022e-07, 1.412113e-07])
larg_myf3d3 = np.array([9.699560e-01, 2.435111e+00, 3.898078e+00, 5.360646e+00, 6.823072e+00, 8.285431e+00, 9.747753e+00])

larg_theo_myf3d3 = partial_widths(lamF3d3x3, myf3d3, MXs)

plt.figure()
plt.plot(myf3d3, larg_myf3d3, "co--")
plt.xlabel(r"Masse du mediateur $myf3d3$ [GeV]")
plt.ylabel("Largeur du mediateur wyf3d3 [GeV^-1]")
plt.title("Largeur du mediateur en fonction du couplage")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.errorbar(myf3d3, cs_myf3d3, yerr=cs_myf3d3_err, fmt="o", capsize=3, label="Données")
plt.xlabel(r"Masse du mediateur $myf3d3$ [GeV]")
plt.ylabel(r"Section efficace $\sigma$ [pb]")
plt.title("Section efficace en fonction du couplage")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()
