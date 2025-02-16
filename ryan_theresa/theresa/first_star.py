# python try_1.py 2d testing.cfg

# General imports
import os
import sys
import mc3
import pickle
import starry
import shutil
import subprocess
import progressbar
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Taurex imports
import taurex
from taurex import chemistry
from taurex import planet
from taurex import stellar
from taurex import model
from taurex import pressure
from taurex import temperature
from taurex import cache
from taurex import contributions
from taurex import optimizer
# This import is explicit because it's not included in taurex.temperature. Bug?
from taurex.data.profiles.temperature.temparray import TemperatureArray

# Taurex is a bit...talkative
import taurex.log
taurex.log.disableLogging()


# Directory structure
maindir    = os.path.dirname(os.path.realpath(__file__))
libdir     = os.path.join(maindir, 'lib')
moddir     = os.path.join(libdir,  'modules')
ratedir    = os.path.join(moddir,  'rate')
transitdir = os.path.join(moddir, 'transit')

# Lib imports
sys.path.append(libdir)
import cf
import atm
import my_pca as pca
import star_eigen as eigen
import model
import star_plots as plots
import mkcfg
import star_utils as utils
import constants   as c
import star_fitclass    as fc
import taurexclass as trc

starry.config.quiet = True

# Starry seems to have a lot of recursion
sys.setrecursionlimit(10000)

def finding_A(cfile):
    # Create the master fit object
    fit = fc.Fit()

    print("Reading the configuration file.")
    fit.read_config(cfile)
    cfg = fit.cfg

    print("Reading the data.")
    fit.read_data()

    lmax = cfg.twod.lmax

    print("First star and planet objects.")
    star = utils.initsystem(fit, 1)

    # print("Computing planet and star positions at observation times.")
    # fit.x, fit.y, fit.z = [a.eval() for a in system.position(fit.t)]

    print("Finding Inclination of planet")
    print(cfg.star.inc)

    # planet.map.inc = 88.0

    print("Finding Design Matrix")
    A = star.map.design_matrix(theta=np.linspace(0,360,1000)) # how do I decide the angular phase???
    print(A.eval())

    print("Finding Rank of Design Matrix")

    nsamples = cfg.twod.nsamples

    # print(nsamples, lmax)

    R = np.empty((nsamples, lmax))

    for k in range(nsamples):
        
        R[k] = [
            np.linalg.matrix_rank(A.eval()[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
        ]
        # print(R[k])
    
    R = np.median(R, axis=0)   

    print(R)

    print("Finding Eigencurves")

    #COME BACK TO THIS - I think I don't need the y00 flux since don't need to worry about eclipse period?

    # fit.sflux_y00 = np.zeros(fit.t.shape)

    #may not need this a pflux_y00 looks like all 1s (see above)
    # print("Calculating uniform-map planet and star fluxes.")
    # fit.sflux= [a.eval() for a in  \
    #                             system.flux(fit.t, total=True)]

    # Uniform star object
    star = utils.initsystem(fit, 1)

    fit.sflux = star.map.flux(theta=fit.t.shape).eval()
    
    print(f"The sflux is: {fit.sflux}")

    print(f"The star map has flux {star.map.flux(theta=np.linspace(0,360,len(fit.t))).eval()}")

    print(f"The time shape is {fit.t.shape}")

    print(star.map.inc.eval())

    star.map.show(theta=np.linspace(0, 360, 50))

  

    fit.maps = []

    fit.maps.append(fc.Map())
    m = fit.maps[0]
    m.ncurves = cfg.twod.ncurves
    m.lmax    = cfg.twod.lmax
    # m.wlmid = fit.wlmid[i]

    if not os.path.isdir(cfg.outdir):
        os.mkdir(cfg.outdir)

    m.subdir = 'filt{}'.format(1)
    outdir = os.path.join(cfg.outdir, m.subdir)

    if not os.path.isdir(os.path.join(cfg.outdir, m.subdir)):
        os.mkdir(os.path.join(cfg.outdir, m.subdir))

    # New planet object with updated lmax
    star = utils.initsystem(fit, m.lmax)

    print("Running PCA to determine eigencurves.")
    m.eigeny, m.evalues, m.evectors, m.ecurves, m.lcs = \
    eigen.mkcurves(star, fit.t, m.lmax, fit.sflux,
                    ncurves=m.ncurves, method=cfg.twod.pca)
    
    # print("Stopping")
    # return None
    
    print("Re-Calc Design Matrix & Rank")
    # new_A = new_planet.map.design_matrix(theta=np.linspace(0,360,1000)) 
    # how do I decide the angular phase??? - It is just the number of rows you want for each vector
    # print(A.eval())

    # planet_A = new_planet.map.design_matrix(theta=np.linspace(0,360,1000))

    # planet_R = np.empty((nsamples, lmax))

    # for k in range(nsamples):
        
    #     planet_R[k] = [
    #         np.linalg.matrix_rank(planet_A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
    #     ]
    #     # print(R[k])
    
    # planet_R = np.median(planet_R, axis=0)   

    # print(f"The new rank of the planet is {planet_R}")



    # print(f"The evectors value looks like {m.evectors} with shape {m.evectors.shape}\n")

    print(f"The lcs value looks like {m.lcs} with shape {m.lcs.shape}\n")

    print(f"The ecurves (proj) value looks like {m.ecurves} with shape {m.ecurves.shape}\n")

    # print(f"The eigeny value looks like {m.eigeny} with shape {m.eigeny.shape}\n")

    # print(f"The evalues value looks like {m.evalues} with shape {m.evalues.shape}\n")


    # lcs_A = m.lcs

    # lcs_R = np.empty((nsamples, lmax))

    # for k in range(nsamples):
        
    #     lcs_R[k] = [
    #         np.linalg.matrix_rank(lcs_A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
    #     ]
    #     # print(R[k])
    
    # lcs_R = np.median(lcs_R, axis=0)   

    # print(f"The new rank with lcs is {lcs_R}")

    lcs_A_T = m.lcs.T

    lcs_R_T = np.empty((nsamples, lmax*2))

    for k in range(nsamples):
        
        lcs_R_T[k] = [
            np.linalg.matrix_rank(lcs_A_T[:, : (l + 1) ** 2]) for l in range(1, lmax*2 + 1)
        ]
        # print(R[k])
    
    lcs_R_T = np.median(lcs_R_T, axis=0)   

    print(f"The new rank with lcs.T is {lcs_R_T}")


    # evect_A = m.ecurves

    # evect_R = np.empty((nsamples, lmax))

    # for k in range(nsamples):
        
    #     evect_R[k] = [
    #         np.linalg.matrix_rank(evect_A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
    #     ]
    #     # print(R[k])
    
    # evect_R = np.median(evect_R, axis=0)   

    # print(f"The new rank with ecurves is {evect_R}")

    evect_A_T = m.ecurves.T

    evect_R_T = np.empty((nsamples, lmax))

    for k in range(nsamples):
        
        evect_R_T[k] = [
            np.linalg.matrix_rank(evect_A_T[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
        ]
        # print(R[k])
    
    evect_R_T = np.median(evect_R_T, axis=0)   

    print(f"The new rank with ecurves.T is {evect_R_T}")

    eigeny_A = m.eigeny #m.eigeny.design_matrix()

    eign_R_1 = np.empty((nsamples, lmax))

    for k in range(nsamples):
        
        eign_R_1[k] = [
            np.linalg.matrix_rank(eigeny_A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
        ]
        # print(R[k])
    
    eign_R_1 = np.median(eign_R_1, axis=0)

    print(f"The new rank with eigeny is {eign_R_1}")

    eigeny_A_T = m.eigeny #m.eigeny.design_matrix()

    eign_R_T = np.empty((nsamples, lmax))

    for k in range(nsamples):
        
        eign_R_T[k] = [
            np.linalg.matrix_rank(eigeny_A_T[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
        ]
        # print(R[k])
    
    eign_R_T = np.median(eign_R_T, axis=0)

    print(f"The new rank with eigeny.T is {eign_R_T}")


    # print(f"The designe matrix of eigeny is {eigeny_A}.\nIt has rank {eigen_R_2}")
    # plots.lightcurves(fit.t, m.lcs, outdir)

    # plots.eigencurves(fit.t, m.ecurves, outdir, "ecurves_curves", ncurves=m.ncurves)
    

    # plots.eigencurves(fit.t, m.lcs, outdir, "lcs_curves", ncurves=m.ncurves)

    plots.lightcurves(fit.t, m.lcs, outdir, "lcs_light_curves")

    



    plots.emaps(star, m.eigeny, outdir, proj='rect')
    #This creates the star maps I was seeing in the line below

    # star.map.show(theta=np.linspace(0, 360, 50))


    
    print("done")








if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 3:
        print("ERROR: Call structure is run.py <mode> <configuration file>.")
        sys.exit()
    else:
        mode  = sys.argv[1]
        cfile = sys.argv[2]

    if mode in ['2d', '2D']:
        finding_A(cfile)
    # elif mode in ['3d', '3D']:
    #     # Read config to find location of output, load output,
    #     # then read config again to get any changes from 2d run.
    #     fit = fc.Fit()
    #     fit.read_config(cfile)
    #     fit = fc.load(outdir=fit.cfg.outdir)
    #     fit.read_config(cfile)
    #     # 3D mapping doesn't care about the degree of harmonics, so
    #     # just use 1
    #     star, planet, system = utils.initsystem(fit, 1)
    #     map3d(fit, system)
    else:
        print("ERROR: Unrecognized mode. Options are <2d, 3d>.")
