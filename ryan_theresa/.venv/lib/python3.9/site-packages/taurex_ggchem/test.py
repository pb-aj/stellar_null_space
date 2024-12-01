from taurex_ggchem.chemistry.ggchemistry import GGChem 
import taurex_ggchem.external.fort_ggchem as fchem
from taurex.constants import AMU
from taurex.cache import OpacityCache
import sys
sys.path.insert(0,'./taurex_ggchem')
OpacityCache().set_opacity_path('/Users/ahmed/Documents/taurex_files/taurex_cobweb/Input/xsec/TauRex_sampled_xsecs_R10000_0.3-15')

dispol_filorino = ['/Users/ahmed/Documents/repos/GGchem/data/dispol_BarklemCollet.dat'.ljust(200),
                   '/Users/ahmed/Documents/repos/GGchem/data/dispol_StockKitzmann_withoutTsuji.dat'.ljust(200), 
                   '/Users/ahmed/Documents/repos/GGchem/data/dispol_WoitkeRefit.dat'.ljust(200),'']


elements = [s.strip() for s in [' H','He','Li','Be',' B',' C',' N',' O',' F','Ne','Na','Mg','Al','Si',' P',' S',
            'Cl','Ar',' K','Ca','Sc','Ti',' V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As',
            'Se','Br','Kr','Rb','Sr',' Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te',
            ' I','Xe','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
            'Hf','Ta',' W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Th',' U']]

abundances = [9.271E-01,7.159E-02,1.077E-11,1.382E-11,2.305E-10,3.112E-04,8.895E-05,7.008E-04,3.279E-08,6.174E-05,
              2.168E-06,3.588E-05,2.771E-06,3.992E-05,2.816E-07,1.554E-05,2.811E-07,2.183E-06,1.275E-07,2.176E-06,
              1.109E-09,1.041E-07,9.783E-09,4.792E-07,2.268E-07,2.231E-05,8.456E-08,1.698E-06,1.372E-08,3.810E-08,
              7.148E-10,3.430E-09,0.000E+00,0.000E+00,0.000E+00,0.000E+00,4.373E-10,7.110E-10,1.401E-10,5.463E-10,
              5.364E-11,1.169E-10,6.163E-11,2.421E-11,3.512E-11,1.155E-11,6.650E-11,4.340E-11,9.446E-11,1.023E-11,
              0.000E+00,0.000E+00,0.000E+00,7.499E-11,9.072E-11,1.794E-11,3.557E-11,8.842E-12,2.591E-11,8.286E-12,
              4.099E-12,1.585E-11,7.839E-13,1.533E-11,0.000E+00,7.449E-12,1.475E-12,7.200E-12,7.121E-12,6.980E-12,
              0.000E+00,2.711E-11,6.691E-13,1.310E-11,1.296E-11,5.748E-11,6.325E-05,1.242E-10,6.096E-12,6.013E-11,
              5.962E-11,1.611E-12,5.234E-12]

selected = ['H', 'He', 'C', 'N', 'O', 'Na', 'Mg', 'Si', 'Fe', 'Al', 'Ca', 'Ti', 'S', 'Cl', 'K', 'Li', 'F', 'P', 'V', 'Cr', 'Mn', 'Ni', 'Zr', 'W', 'el']
sel_ab = [abundances[elements.index(s)] for s in selected if s is not 'el']

fchem.parameters.initchem_info = True

gg = GGChem(abundance_profile='solar',equilibrium_condensation=False,Tfast=1000.0) 

#print(fchem.parameters.elements)
#fchem.parameters.verbose = 2
#quit()
#print(gg._molecules)
import numpy as np 
import math
tp = np.logspace(math.log10(2500),math.log10(100),100) 

pp = np.ones(100)*1e5
pp 
gg.initialize_chemistry(nlayers=100,temperature_profile=tp,pressure_profile=pp) 
#print(gg._elements)
#quit()
#print(fchem.dust_data.eps0)
#print(fchem.dust_data.eps0.dtype)

#print(12+np.log10(fchem.dust_data.eps0))
nmol = fchem.chemistry.nmole 
nelem = fchem.dust_data.nelem 
print(gg.muProfile/AMU)
#print(tp)
