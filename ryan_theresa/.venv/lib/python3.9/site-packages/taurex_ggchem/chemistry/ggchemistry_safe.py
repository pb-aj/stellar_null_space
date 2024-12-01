from taurex.chemistry import Chemistry
import numpy as np
import taurex_ggchem.external.fort_ggchem as fchem 
import os
from taurex.core import fitparam
#from taurex.util.fortran import FortranSafeCaller
import pkg_resources
from taurex.util.fortran import SafeFortranCaller, FortranStopException
from taurex.exceptions import InvalidModelException

selected = ['H','He','C', 'N', 'O', 'Na', 'Mg', 'Si', 'Fe', 'Al', 'Ca', 'Ti', 'S', 'Cl', 'K', 'Li', 'F', 'P', 'V', 'Cr', 'Mn', 'Ni', 'Zr', 'W']

class GGChem(Chemistry):



    allowed_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 
                        'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 
                        'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'W']

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
                 new_pre_method=2, 
                 new_full_it=False, 
                 new_fast_level=1):
        super().__init__(self.__class__.__name__)

        self._init_dispol_files = dispol_files
        self._init_abundance_profile = abundance_profile
        self._init_selected_elements = selected_elements
        self._init_ratio_elements = ratio_elements
        self._init_ratios_to_O = ratios_to_O
        self._init_he_h_ratio = he_h_ratio
        self._init_metallicity = metallicity
        self._init_include_charge = include_charge
        self._init_equilibrium_condensation = equilibrium_condensation
        self._init_dustchem_file = dustchem_file
        self._init_Tfast = Tfast
        self._init_new_back_it = new_back_it
        self._init_new_back_fac = new_back_fac
        self._init_new_pre_method = new_pre_method
        self._init_new_full_it = new_full_it
        self._init_new_fast_level = new_fast_level

        self._safe_caller = SafeFortranCaller(fchem.__name__,logger=self)
        self.reinitialize_ggchem()
        self.restore_molecule_names()
        self._setup_active_inactive()
        self.add_ratio_params()
    def reinitialize_ggchem(self):

        dispol_files = self._init_dispol_files
        abundance_profile = self._init_abundance_profile
        selected_elements = self._init_selected_elements
        ratio_elements = self._init_ratio_elements
        ratios_to_O = self._init_ratios_to_O
        he_h_ratio = self._init_he_h_ratio
        metallicity = self._init_metallicity
        include_charge = self._init_include_charge
        equilibrium_condensation = self._init_equilibrium_condensation
        dustchem_file = self._init_dustchem_file
        Tfast = self._init_Tfast
        new_back_it = self._init_new_back_it
        new_back_fac = self._init_new_back_fac
        new_pre_method = self._init_new_pre_method
        new_full_it = self._init_new_full_it
        new_fast_level = self._init_new_fast_level


        elements = [s.strip() for s in selected_elements]
        if 'O' not in elements:
            elements.append('O')
        if 'He' not in elements:
            elements.insert(0,'He')
        if 'H' not in elements:
            elements.insert(0,'H')

        


        self._he_h_ratio = he_h_ratio
        self._metallicity = metallicity
        self._base_data_path = pkg_resources.resource_filename('taurex_ggchem','external/data')

        self.info('Using GGChem chemistry')

        ignored_elements = [s.strip() for s in elements if s not in self.allowed_elements]

        if len(ignored_elements) > 0:
            self.info('Elements ignored: %s',ignored_elements)

            


        elements = [s.strip() for s in elements if s in self.allowed_elements]
        if include_charge:
            self._charge = include_charge
            elements.append('el')
            self.info('Ions activated')
        else:
            self._charge = False
        self._safe_caller.set_val('chemistry.newfullit',new_full_it)
        self._safe_caller.set_val('chemistry.newbackit',new_back_it)
        self._safe_caller.set_val('chemistry.newbackfac',new_back_fac)
        self._safe_caller.set_val('chemistry.newfastlevel',new_fast_level)
        self._safe_caller.set_val('chemistry.newpremethod',new_pre_method)



        self._passed_elements = elements
        self._elements = elements
        self._safe_caller.set_val('parameters.verbose',0)
        self._initial_abundances = self._safe_caller.call('fort_ggchem.init_lean')



        self._safe_caller.set_val('parameters.initchem_info', False)

        if dispol_files is None:
            dispol_files = [os.path.join(self._base_data_path,s) for s in ['dispol_BarklemCollet.dat',
                                                                           'dispol_StockKitzmann_withoutTsuji.dat',
                                                                           'dispol_WoitkeRefit.dat',
                                                                    ]]                        
        self._safe_caller.set_val('parameters.elements', ' '.join([s.ljust(2) for s in elements]).ljust(200))
        #print('ELEMENTS',fchem.parameters.elements)
        self.info('Elements in system: %s',elements)
        dispol = [s.ljust(200) for s in dispol_files]  


        self.info('Initializing GGchemistry')
        self._safe_caller.call('fort_ggchem.init_taurex_chemistry',dispol,self._charge)
        
        self.setup_abundances(abundance_profile)
        self.setup_ratios(ratio_elements,ratios_to_O)

        self._safe_caller.set_val('parameters.model_eqcond',equilibrium_condensation)

        self._safe_caller.set_val('parameters.tfast',Tfast)

        self.info('T-fast %s K',Tfast)
        self.info('Eq. Condensation: %s', equilibrium_condensation)

        nmole = self._safe_caller.get_val('chemistry.nmole')

        _mols = self._safe_caller.call('fort_ggchem.copy_molecule_names',nmole) 
        mols = np.lib.stride_tricks.as_strided(_mols, strides=(_mols.shape[1],1))

        mols = [s[0].decode('utf-8') for s in np.char.strip(mols.view('S20'))]

        self._molecules = mols




        if dustchem_file is None:
            dustchem_file = os.path.join(self._base_data_path,'DustChem.dat')

        self.info('Loading dustchem file %s',dustchem_file)
        self._safe_caller.set_val('dust_data.dustchem_file', dustchem_file.ljust(200))
        self._safe_caller.call('init_dustchem_taurex')

        ndust = self._safe_caller.get_val('dust_data.ndust')

        _dust = self._safe_caller.call('fort_ggchem.copy_dust_names',ndust) 
        dust = np.lib.stride_tricks.as_strided(_dust, strides=(_dust.shape[1],1))

        dust = [s[0].decode('utf-8') for s in np.char.strip(dust.view('S20'))]

        self._condensates = dust


        #quit()
        # self.update_abundances()

    @property
    def condensates(self):
        """
        Returns a list of condensates in the atmosphere.

        Returns
        -------
        active : :obj:`list`
            List of condensates

        """

        return self._condensates


    @property
    def condensateMixProfile(self):
        """
        **Requires implementation**

        Should return profiles of shape ``(ncondensates,nlayers)``.
        """
        if len(self.condensates) == 0:
            return None
        else:
            return self._dust

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
                    self._initial_abundances[self.get_element_index(el)] = ab
        
            
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
        elnums = self._safe_caller.get_val('chemistry.elnum')
        index = self._safe_caller.get_val(f'chemistry.{element.strip().lower()}')


        return elnums[index-1]-1
        #return self.get_element_index(element)


    def _setup_active_inactive(self):

        active_mols = []
        inactive_mols =[]
        active_indices = []
        inactive_indices = []
        nmol = self._safe_caller.get_val('chemistry.nmole')

        clean_elements =[ el if el != 'el' else 'e-' for el in self._elements]

        for idx,m in enumerate(clean_elements + self._molecules):
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



        #print(ab)
        #self.reinitialize_ggchem()
        ab = self.update_abundances()
        #fchem.structure.tgas[:nlayers] = temperature_profile
        #fchem.structure.press[:nlayers] = pressure_profile*10 # to dyn/cm2
        self.info('Running GGChem equilibrium code...')
        nmol = self._safe_caller.get_val('chemistry.nmole')
        nelem = self._safe_caller.get_val('chemistry.nelm')
        ndust = self._safe_caller.get_val('dust_data.ndust')
        mols = []
        dust = []
        for t,p in zip(temperature_profile,pressure_profile):
            try:
                m,d = self._safe_caller.call('fort_ggchem.run_ggchem',nelem+nmol,ndust,t,p*10,ab)
                mols.append(m)
                dust.append(d)
            except FortranStopException:
                #self.warning('Error occured in Z:%s ratios:%s T:%s P:%s',self._metallicity,self._ratios,t,p)
                self._safe_caller.cleanup()
                self.reinitialize_ggchem()
                raise InvalidModelException('GGChem most likely STOPPED due to a error')
        self._mols = np.stack(mols).T
        self._dust = np.stack(dust).T
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
        return ['ggchem', ]

    BIBTEX_ENTRIES = [
        """
        @article{ ggchem,
            author = {{Woitke, P.} and {Helling, Ch.} and {Hunter, G. H.} and {Millard, J. D.} and {Turner, G. E.} and {Worters, M.} and {Blecic, J.} and {Stock, J. W.}},
            title = {Equilibrium chemistry down to 100 K - Impact of silicates and phyllosilicates on the carbon to oxygen ratio},
            DOI= "10.1051/0004-6361/201732193",
            url= "https://doi.org/10.1051/0004-6361/201732193",
            journal = {A\&A},
            year = 2018,
            volume = 614,
            pages = "A1",
        }
        """
    ]