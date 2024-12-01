from taurex.chemistry import Chemistry
import numpy as np
import taurex_ggchem.external.fort_ggchem as fchem 
import os
from taurex.core import fitparam
import pkg_resources
from taurex.util.fortran import stdout_redirector
selected = ['H','He','C', 'N', 'O', 'Na', 'Mg', 'Si', 'Fe', 'Al', 'Ca', 'Ti', 'S', 'Cl', 'K', 'Li', 'F', 'P', 'V', 'Cr', 'Mn', 'Ni', 'Zr', 'W']


class StreamToLogger:

    def __init__(self, logger):
        self.logger = logger

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            out = line.rstrip()

            if isinstance(out, bytes):
                out = out.decode('utf-8')
            self.logger.debug(buf)


class GGChemNonSafe(Chemistry):



    allowed_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 
                        'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 
                        'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'W',]

    element_index_dict = dict(zip(allowed_elements[1:],range(0,len(allowed_elements))))

    def __init__(self, dispol_files = None, 
                 abundance_profile=None,
                 selected_elements = selected,
                 ratio_elements=None,
                 ratios_to_O=None,
                 he_h_ratio=0.083,
                 metallicity=1.0,
                 include_charge = False,
                 equilibrium_condensation=False, 
                 dustchem_file=None,
                 Tfast=1000,
                 new_back_it=6,  
                 new_back_fac=1e5,
                 new_pre_method=2, new_full_it=False, new_fast_level=1):
        super().__init__(self.__class__.__name__)


        self._stream_to_logger = StreamToLogger(self)

        elements = [s.strip() for s in selected_elements]
        if 'O' not in elements:
            elements.append('O')
        if 'He' not in elements:
            elements.insert(0,'He')
        if 'H' not in elements:
            elements.insert(0,'H')

        if include_charge:
            elements.append('el')
         
        self._he_h_ratio = he_h_ratio
        self._metallicity = metallicity
        self._base_data_path = pkg_resources.resource_filename('taurex_ggchem','external/data')

        self._charge = 'el' in elements

        self.info('Using GGChem chemistry')

        ignored_elements = [s.strip() for s in elements if s not in self.allowed_elements]

        if len(ignored_elements) > 0:
            self.info('Elements ignored: %s',ignored_elements)

        if self._charge:
            self.info('Ions activated')


        elements = [s.strip() for s in elements if s in self.allowed_elements]


        fchem.chemistry.newfullit    = new_full_it
        fchem.chemistry.newbackit    = new_back_it
        fchem.chemistry.newbackfac   = new_back_fac
        fchem.chemistry.newfastlevel = new_fast_level
        fchem.chemistry.newpremethod = new_pre_method



        self._passed_elements = elements
        self._elements = elements
        fchem.parameters.verbose = 0
        self._initial_abundances = fchem.fort_ggchem.init_lean()



        fchem.parameters.initchem_info = False

        if dispol_files is None:
            dispol_files = [os.path.join(self._base_data_path,s) for s in ['dispol_BarklemCollet.dat',
                                                                           'dispol_StockKitzmann_withoutTsuji.dat',
                                                                           'dispol_WoitkeRefit.dat',
                                                                    ]]
                                                            
        fchem.parameters.elements = ' '.join([s.ljust(2) for s in elements]).ljust(200)
        #print('ELEMENTS',fchem.parameters.elements)
        self.info('Elements in system: %s',elements)
        dispol = [s.ljust(200) for s in dispol_files]  


        self.info('Initializing GGchemistry')
        fchem.fort_ggchem.init_taurex_chemistry(dispol,self._charge)
        #print(fchem.chemistry.nelm)
        #quit()


        # _atms = fchem.fort_ggchem.copy_atom_names(fchem.chemistry.nelm) 
        # atms = np.lib.stride_tricks.as_strided(_atms, strides=(_atms.shape[1],1))

        # atms = [s[0].decode('utf-8') for s in np.char.strip(atms.view('S2'))]

        # self._elements = atms
        # self._passed_elements = atms
        
        self.setup_abundances(abundance_profile)
        self.setup_ratios(ratio_elements,ratios_to_O)

        #quit()
        # self.update_abundances()
        fchem.parameters.model_eqcond = equilibrium_condensation

        fchem.parameters.tfast = Tfast

        self.info('T-fast %s K',Tfast)
        self.info('Eq. Condensation: %s', equilibrium_condensation)

        _mols = fchem.fort_ggchem.copy_molecule_names(fchem.chemistry.nmole) 
        mols = np.lib.stride_tricks.as_strided(_mols, strides=(_mols.shape[1],1))

        mols = [s[0].decode('utf-8') for s in np.char.strip(mols.view('S20'))]

        self._molecules = mols


        self.restore_molecule_names()

        if dustchem_file is None:
            dustchem_file = os.path.join(self._base_data_path,'DustChem.dat')

        self.info('Loading dustchem file %s',dustchem_file)
        fchem.dust_data.dustchem_file = dustchem_file.ljust(200)   
        fchem.init_dustchem_taurex()

        #quit()
        # self.update_abundances()

        self._setup_active_inactive()
        self.add_ratio_params()

    def setup_abundances(self, profile):
        
        if profile is not None:
            self.info('Selected abundance profile %s',profile)
            abundance_file = os.path.join(self._base_data_path,'Abundances.dat')
            chosen = 4 # 4 - earth 5 - ocean 6 - solar 7- meteorite
            profile = profile.lower()

            if profile in ('earthcrust',):
                chosen = 4
            elif profile in ('ocean',):
                chosen = 5
            elif profile in ('sun', 'star','solar',):
                chosen = 6
            elif profile in ('meteor', 'meteorites',):
                chosen = 7

            elements = []
            abundances = np.zeros(shape=(len(self.allowed_elements),))

            with open(abundance_file,'r') as f:
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                for line in f:
                    split_line = line.split()
                    el = split_line[2].strip()

                    if el not in self.allowed_elements:
                        continue
                    
                    elements.append(el)
                    ab = float(split_line[chosen])
                    self.info('%s %s', el, ab)
                    abundances[self.get_element_index(el)] = ab
        
        #self.update_abundances(elements=elements,abundances=abundances)
        self._initial_abundances/=self._initial_abundances[0]
        self.info('H relative abundances.....')
        for el,ab in zip(self.allowed_elements,self._initial_abundances):
            self.info('%s %s', el, ab)
            
    
        #raise
    def setup_ratios(self,ratio_elements,ratios_to_O):
        self._metal_idx = [self.get_element_index(e) for e in self._elements if e not in ('H', 'He','el',)]
        self._metal_elements = [e for e in self._elements if e not in ('H', 'He','el',)]
        self.info('Metals detected %s',list(zip(self._metal_idx,self._metal_elements)))
        O_idx = self.get_element_index('O')
        self._O_abund = self._initial_abundances[O_idx]
        self._ratios = self._initial_abundances[self._metal_idx]
        self._ratios/= self._O_abund


        if ratio_elements is not None:
            for relm,rat in zip(ratio_elements,ratios_to_O):
                index = self._metal_elements.index(relm)
                self._ratios[index] = rat

        self.info('Metallic ratios against O')
        for met, rat in zip(self._metal_elements,self._ratios):
            self.info('%s %s',met, rat)

    def get_element_index(self, element):
        return fchem.chemistry.elnum[int(getattr(fchem.chemistry,element.strip().lower()))-1]-1
        #return self.get_element_index(element)


    def _setup_active_inactive(self):

        active_mols = []
        inactive_mols =[]
        active_indices = []
        inactive_indices = []
        nmol = fchem.chemistry.nmole 

        for idx,m in enumerate(self._elements + self._molecules):
            if m in self.availableActive:
                active_mols.append(m)
                active_indices.append(idx)
            else:
                inactive_mols.append(m)
                inactive_indices.append(idx)              

        self.info('Active molecules %s',active_mols)
        self.info('Inactive molecules %s',inactive_mols)

        self._active_index = np.array(active_indices,dtype=np.integer)
        self._inactive_index = np.array(inactive_indices,dtype=np.integer)

        self._active_mols = active_mols
        self._inactive_mols = inactive_mols

    
    def restore_molecule_names(self):
        import re
        elm = [s for s in self._passed_elements if len(s)>1]
        elem_upp = [s.upper() for s in self._passed_elements if len(s)>1]

        rep = dict(zip(elem_upp, elm))

        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        pattern = re.compile("|".join(rep.keys()))

        self._molecules = [pattern.sub(lambda m: rep[re.escape(m.group(0))], s) for s in self._molecules]


    def update_abundances(self):
        
        abundances = self._initial_abundances
        ratios = self._ratios
        O_abund = self._O_abund*self._metallicity
        abundances[self._metal_idx]=ratios*O_abund
        abundances/=abundances[self.get_element_index('H')]
        abundances[self.get_element_index('He')] = self._he_h_ratio

        return abundances
        



    def initialize_chemistry(self, nlayers=100, temperature_profile=None,
                             pressure_profile=None, altitude_profile=None):
        from taurex.constants import KBOLTZ
        ab = self.update_abundances()


        #print(ab)

        #fchem.structure.tgas[:nlayers] = temperature_profile
        #fchem.structure.press[:nlayers] = pressure_profile*10 # to dyn/cm2
        self.info('Running GGChem equilibrium code...')
        nmol = fchem.chemistry.nmole 
        nelem = fchem.chemistry.nelm 
        with stdout_redirector(self._stream_to_logger):
            mols = [fchem.fort_ggchem.run_ggchem(nelem+nmol,t,p*10,ab) for t,p in zip(temperature_profile,pressure_profile) ]
        self._mols = np.stack(mols).T
        #self._vmr = mols/np.sum(mols,axis=0)
        #self._vmr = self._mols
        #self._vmr = mols/1e-6 #m-3
        self._vmr =self._mols#/= pressure_profile/(KBOLTZ*temperature_profile)
        self.info('Computing mu Profile')
        self.compute_mu_profile(nlayers)


    @fitparam(param_name='metallicity',param_latex='Z',default_bounds=[0.2,2.0],default_fit=False)
    def metallicity(self):
        return self._metallicity
    
    @metallicity.setter
    def metallicity(self,value):
        self._metallicity = value

    def add_ratio_params(self):

        for idx,element in enumerate(self._metal_elements):
            if element == 'O':
                continue
            param_name = f'{element}_O_ratio'
            param_tex = f'{element}/O'
            id_val = self._metal_elements.index(element)
            def read_mol(self, idx=id_val):
                return self._ratios[idx]

            def write_mol(self, value, idx=id_val):
                self._ratios[idx] = value

            fget = read_mol
            fset = write_mol

            bounds = [1.0e-12, 0.1]

            default_fit = False
            self.add_fittable_param(param_name, param_tex, fget,
                                    fset, 'linear', default_fit, bounds)


    @property
    def activeGases(self):
        return self._active_mols

    @property
    def inactiveGases(self):
        return self._inactive_mols

    @property
    def activeGasMixProfile(self):
        return self._vmr[self._active_index]

    @property
    def inactiveGasMixProfile(self):
        return self._vmr[self._inactive_index]

    def ratio(self, element_a, element_b):
        element_a = element_a.strip()
        element_b = element_b.strip()

        element_a_idx = self.get_element_index(element_a)
        element_b_idx = self.get_element_index(element_b)
        return float(fchem.dust_data.eps0[element_a_idx]/fchem.dust_data.eps0[element_b_idx])

    @classmethod
    def input_keywords(cls):
        return ['ggchem-nonsafe', ]