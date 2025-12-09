from parameters import *
import kinetics    as ki
import equilibrium as eq
import transport   as tr

class WeatheringFluxInterpolator:
    def __init__(self, rock_type='bash'):
        self.rock_type = rock_type
        print(f"Initializing Weathering Interpolator for {rock_type}...")
        
        self.grid_PCO2 = np.logspace(-5, 0, 12)
        self.grid_T    = np.linspace(275, 350, 12)
        self.grid_P    = np.array([1.0])           
        
        self.grid_L    = np.array([1.0])          
        self.grid_tsoil = np.logspace(0, 7, 8)

        print("  - Loading Databases...")
        self.KeqFuncs = eq.import_thermo_data('./database/species.csv')
        self.logkDict = ki.import_kinetics_data()

        print("  - Building Equilibrium Grid...")
        self.DICeqFuncs = eq.get_DICeq(self.grid_PCO2, self.grid_T, self.grid_P, self.KeqFuncs)
        self.interp_Ceq = self.DICeqFuncs[self.rock_type]['HCO3']

        print("  - Building Kinetics Grid...")
        self.kinetics_T  = np.linspace(250, 400, 20) 
        self.kinetics_pH = np.linspace(-5, 20, 50)
        self.kFuncs = ki.get_keff(self.kinetics_T, self.kinetics_pH, self.logkDict)

        test_val = self.interp_Ceq(np.array([[280e-6, 288, 1.0]]))
        print(test_val)

if __name__ == "__main__":
    # Select rock type here
    flux_model = WeatheringFluxInterpolator()