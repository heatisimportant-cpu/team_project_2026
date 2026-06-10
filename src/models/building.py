import numpy as np
from utils.constants import (
    RHO_WATER, RHO_AIR, C_WATER_SPEC, C_AIR_SPEC,
    C_INT_SPEC, H_UFH_SPEC, H_UFH_SURF_SPEC, V_TS_SPEC, V_UFH_SPEC
)


class Building:
    """
   
    With the building class, a bulding model can be generated from pysical and geometrical parameters as specified in the data/buildings directory.
    Different calculation methods are available to model the heat flows of the building and an underfloor heating system in a dynamic way (2R2C, 4R3C, 5R4C).
    Ordinary differential equations (ODEs) are used to model the respective thermal networks, containing different thermal resistances (R) and capacities (C).
    The ODEs are transformed into state space equations in the form :math:`\\dot{x} = A \cdot x + B \cdot u` and :math:`y = C \cdot x + D \cdot u` with:
        - A : System matrix
        - B : Control matrix
        - C : Output matrix
        - D : Feedthrough matrix
        - x : state vector
        - u : input vector (containing control and disturbance variables)
        - y : output vector

    For more details about the individual calculation methods check out the documentation of the individual calc_xRxC functions.
    
    The required input parameters for the chosen calculation method are computed while initializing the building model,
    using the input parameters of the individual building and the following general constants:
        - C_INT_SPEC = 10000 J/m^2/K    : Spec heat capacitiy of interior acc. to ISO52016 Tab. B17
        - H_UFH_SPEC = 4.4 W/m^2/K      : Spec heat transmission coefficient for underfloor heating acc. to `Daniel Rüdiser <https://www.htflux.com/en/dynamic-simulation-and-comparison-of-two-underfloor-heating-systems/>`_
        - H_UFH_SURF_SPEC = 10.8 W/m^2/K : Spec heat convection from floor to air with underfloor heating system acc to EN 1264-5
        - V_TS_SPEC = 5 l/m^2           : Spec volume of thermal heat storage per heated floor area acc. to DGS – German Society for Solar Energy. Guide to Solar Thermal Systems – Expert Presentation on DVD, 9th Edition
        - V_UFH_SPEC = 1.5 l/m^2        : Spec volume of water in underfloor heating system per heated floor area acc. to `BaCoGa Technik GmBH <https://www.bacoga.com/wp-content/uploads/2013/02/Volumenberechnung.pdf>`_
        - R_SI = 0.13 m^2*K/W           : Heat resistance of internal surfaces acc. to DIN EN ISO 6946 Tabelle 7. 
        - h_tr_int = 9.1 W/(m^2K)       : Heattransfer coefficient between building mass and indoor surface ISO13790 12.2.2
        - h_air2surf = 3.45 W/(m^2K)    : Heattransfer coefficient between indoor surface and indoor air ISO13790 7.2.2.2
        - c_P_screed = 1000 # specific heat capacity of screed [J/(kgK)] source: https://www.schweizer-fn.de/stoff/wkapazitaet/wkapazitaet_baustoff_erde.php
        - d_screed = 50     # thickness of screed for UFH [mm] source: https://www.kesselheld.de/heizestrich/
        - rho_screed = 2000 # dry bulk density of screed [kg/m^3] source: https://www.profibaustoffe.com/wp-content/files/TD_2059_2056_ESTRICH-CT-C20-F4-E225-MIT-FASERN_041119.pdf

    Attributes
    ----------
   
    params : dict
        Containing easily accissible building parameters as specified in buildings.py
        
    mdot_hp : float (optional)
        :math:`\dot{m}_{hp}`  : Mass flow rate of the hp system, default = 0.25 [kg/s]
        
    T_room_set_lower : float (optional)
        :math:`T_{room,set,lower}`  : Lower set point temperature, default = 18 [°C]
    
    T_room_set_upper : float (optional)
        :math:`T_{room,set,upper}`  : Upper set point temperature, default = 26 [°C]    
    
    method : string
        Choose calculation method ['2R2C' (default), '4R3C', '5R4C', '6R4C'].
        The electable calc method depends on the provided building parameters.
        The required parameters are described for each calculation method separately.
    
    usage : string
        Choose usage of the building ['Office', 'ResidentialDetached' (default), 'ResidentialFlat', 'SchoolClassroom'].

    verbose: bool
        flag to print and plot further building information

    Notes
    -----
        The following parameters are calculated and added to the params dictionary by initialization. 

            - :math:`H_{ve,tr} = H_{ve} + H_{tr}`   (H_ve_tr)      : Heat transfer coefficient for ventilation and transmission (indoors --> ambient) [W/K]
            - :math:`H_{ve}`                        (H_ve)         : Heat transfer coefficient for ventilation (indoors --> ambient) [W/K]
            - :math:`H_{tr}`                        (H_tr)         : Heat transfer coefficient for transmission (indoors --> ambient) [W/K]
            - :math:`H_{rad,con}`                   (H_rad_con)    : Heat transfer coefficient for radiation and convection (hvac --> indoors) [W/K]
            - :math:`H_{int}`                       (H_int)        : Heat transfer coefficient for convection and radiation (interior surfaces --> indoor air) [W/K]
            - :math:`H_{tr,heavy}`                  (H_tr_heavy)   : Heat transfer coefficient for the heavy building components (wall, floor, roof) (indoors --> ambient) [W/K]
            - :math:`C_{bldg} = C_{wall} + C_{zone}` (C_bldg)      : Heat capacity of the entire building (air and wall) [J/K]
            - :math:`C_{wall}`                      (C_wall)       : Heat capacity of the walls [J/K]
            - :math:`C_{zone} = C_{air} + C_{int}`  (C_zone)       : Heat capacity of the thermal zone [J/K]
            - :math:`C_{air}`                       (C_air)        : Heat capacity of the indoor air [J/K]
            - :math:`C_{int}`                       (C_int)        : Heat capacity of the interior [J/K]
            - :math:`C_{water}`                     (C_water)      : Heat capacity of the water in the hvac system [J/K]
            - :math:`C_{surf}`                      (C_surf)       : Heat capacity fo the inside envelope surface [J/K]
            - :math:`C_{surf_wall}                  (C_surf_wall)  : Heat capacitiy of the wall surface
            - :math:`C_{surf_floor}                 (C_surf_floor) : Heat capacitiy of the floor surface
            - :math:`C_{bldg,heavy}`                (C_bldg_heavy) : Heat capacity of the heavy building components, excluding the inside envelope surface [J/K]
            - :math:`A_{surf}`                      (A_surf)       : Internal surface area [m^2]
            - :math:`A_{mass}`                      (A_mass)       : Internal surface area of the heavy building components [m^2] 
            - :math:`V_{air}`                       (volume_air)   : Indoor air volume [m^3]

    
    Building thermal model using only the 4R3C method.

    States:
    - T_room   : room temperature [°C]
    - T_wall   : wall temperature [°C]
    - T_hp_ret : heating system return temperature [°C]

    Inputs:
    - T_hp_sup    : heating system supply temperature [°C]
    - T_amb       : ambient temperature [°C]
    - Qdot_gains  : internal/solar gains [W]

    """

    def __init__(self,
                 params=None,
                 mdot_hp=0.27,
                 T_room_set_lower=20,
                 T_room_set_upper=24,
                 usage='ResidentialDetached',
                 verbose=False):

        self.params = params
        self.mdot_hp = mdot_hp
        self.T_room_set_lower = T_room_set_lower
        self.T_room_set_upper = T_room_set_upper
        self.usage = usage

        self.__calc_bldg_parameters()
        self.__init_calc_method()

        if verbose:
            self.print_params()
            self.plot_step_response() 

    def __calc_bldg_parameters(self):
        ''' Function to calculate additional building parameters.
        
        Parameters
        ----------
        params : dict
            Containing easily accessible building parameters as specified in docs/buildings/readme.md
        '''
        self.params['volume_air'] = self.params['area_floor'] * self.params['height_room']
        self.params['H_ve_tr'] = self.params['H_ve'] + self.params['H_tr']

        self.params['H_con_floor'] = H_UFH_SURF_SPEC * self.params['area_floor']
        self.params['H_rad_con'] = H_UFH_SPEC * self.params['area_floor']
        self.params['H_tr_floor'] = 1 / (
            (1 / self.params['H_rad_con']) - (1 / self.params['H_con_floor'])
        )

        volume_water = (V_TS_SPEC + V_UFH_SPEC) * self.params['area_floor'] / 1000
        self.params['C_water'] = RHO_WATER * C_WATER_SPEC * volume_water
        self.params['C_air'] = RHO_AIR * C_AIR_SPEC * self.params['volume_air']
        self.params['C_int'] = C_INT_SPEC * self.params['area_floor']
        self.params['C_zone'] = self.params['C_air'] + self.params['C_int']
        self.params['C_bldg'] = self.params['c_bldg'] * self.params['area_floor'] * 3600
        self.params['C_wall'] = self.params['C_bldg'] - self.params['C_zone']

    def __init_calc_method(self):
        """ Set state and input keys according to selected building model building.
        """
        self.state_keys = ("T_room", "T_wall", "T_hp_ret")
        self.input_keys = ("T_hp_sup", "T_amb", "Qdot_gains")

    def calc(self, t, x, args):
        ''' Chooses and calls the building model class depending on selected method.
        
        Parameters
        ----------
        t : not used - but needed for scipy.integrate.solve_ivp() which calls this function

        x : list,
            state vector
                - x[0] : Room temperature [degC]
                - x[i] : depending on selected bldg model [degC]
                - x[-1] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations according to selected building model
        '''
        return self.calc_4r3c(t, x, args[0], args[1:])

    def calc_4r3c(self, t, x, u, p):
        """
        Calculate state changes of building temperature and return flow temperature.

        The following building attributes have to be set:
        H_tr, H_ve, H_rad_con, C_wall, C_zone, C_water

        The corresponding state vector has the following entries:
        [T_room, T_wall, T_hp,ret]
        
        Parameters
        ----------
        x : list
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Wall temperature [degC]
                - x[2] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector 
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations
            - rhs[0] of dT_room/dt   : room temperature [degC/s]
            - rhs[1] of dT_wall/dt   : wall temperature [degC/s]
            - rhs[2] of dT_hp_ret/dt : return flow temperature [degC/s]


        The ordinary differential equations for the 4R3C calculation method are:

        dT_room/dt =
            1/C_zone * (
                Qdot_gains
                + H_rad_con * (T_hp_ret - T_room)
                - 2*H_tr * (T_room - T_wall)
                - H_ve * (T_room - T_amb)
            )

        dT_wall/dt =
            1/C_wall * (
                2*H_tr * (T_room - T_wall)
                - 2*H_tr * (T_wall - T_amb)
            )

        dT_hp_ret/dt =
            1/C_water * (
                mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
                - H_rad_con * (T_hp_ret - T_room)
            )
        """
        T_room = x[0]
        T_wall = x[1]
        T_hp_ret = x[2]

        T_hp_sup = u
        T_amb = p[0]
        Qdot_gains = p[1]

        rhs = np.zeros(3)

        rhs[0] = 1 / self.params['C_zone'] * (
            Qdot_gains
            + self.params['H_rad_con'] * (T_hp_ret - T_room)
            - 2 * self.params['H_tr'] * (T_room - T_wall)
            - self.params['H_ve'] * (T_room - T_amb)
        )

        rhs[1] = 1 / self.params['C_wall'] * (
            2 * self.params['H_tr'] * (T_room - T_wall)
            - 2 * self.params['H_tr'] * (T_wall - T_amb)
        )

        rhs[2] = 1 / self.params['C_water'] * (
            self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret)
            - self.params['H_rad_con'] * (T_hp_ret - T_room)
        )

        return rhs
    
    def calc_casadi(self, x, u, p):
        ''' Chooses and calls the casadi building model class depending on selected method.
        
        Parameters
        ----------
        t : not used - but needed for scipy.integrate.solve_ivp() which calls this function

        x : list,
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations for (casadi.casadi.SX) according to selected building model
        '''

        return self.calc_4r3c_casadi(x, u, p)

    def calc_4r3c_casadi(self, x, u, p):
        ''' This function contains the same building model as the fuction calc_4r3c.
        Here the casadi framework is used to use this model in a MPC controller.
        Check out calc_4r3c for details about the model.
        
        Parameters
        ----------
        x : list,
            state vector
                - x[0] : Room temperature [degC]
                - x[1] : Wall temperature [degC]
                - x[2] : Return flow temperature [degC]
        u : list
            control vector
                - u[0] : Heatpump supply temperature [degC] 
        p : list
            input vector
                - p[0] : Ambient temperature [degC]
                - p[1] : Precomputed gains [W]
        
        Returns
        -------
        List of right hand side equations for (casadi.casadi.SX)
            - rhs[0] of dT_room/dt   : room temperature [degC/s]
            - rhs[1] of dT_wall/dt   : wall temperature [degC/s]
            - rhs[2] of dT_hp_ret/dt : return flow temperature [degC/s]
        '''

        T_room     = x[0] # [degC]
        T_wall     = x[1] # [degC]
        T_hp_ret   = x[2] # [degC]
        T_hp_sup   = u    # [degC]
        T_amb      = p[0] # [degC]
        Qdot_gains = p[1] # [W]

        rhs  = cas.SX.sym("rhs",3)

        rhs[0] = 1 / self.params['C_zone'] * (Qdot_gains + self.params['H_rad_con'] * (T_hp_ret - T_room) \
                - 2 * self.params['H_tr'] * (T_room - T_wall) - self.params['H_ve'] * (T_room - T_amb)) # T_room [degC/s]
        rhs[1] = 1 / self.params['C_wall'] * (2 * self.params['H_tr'] * (T_room - T_wall) \
                - 2 * self.params['H_tr'] * (T_wall - T_amb))     # T_wall [degC/s]
        rhs[2] = 1 / self.params['C_water'] * (self.mdot_hp * C_WATER_SPEC * (T_hp_sup - T_hp_ret) \
                - self.params['H_rad_con'] * (T_hp_ret - T_room)) # T_hp_ret [degC/s]
        return rhs

    def calc_comfort_dev(self, T_room, timestep):
        """ Calculation of the comfort deviation of the building between
        set comfort levels and computed indoor temperature.
        
        Parameters
        ----------
        T_room : numpy.ndarray
            :math:`T_{room}` : Room air temperature  [°C]
            
        timestep : float
            Duration of the integration time step [s]

        Returns
        -------
        dict
            - dev_neg_sum  (float) : sum of negative deviations  [Kh]
            - dev_neg_max  (float) : maximum negative deviation  [K]
            - dev_pos_sum  (float) : sum of positive deviations  [Kh]
            - dev_pos_max  (float) : maximum positive deviation  [K]
        """

        isteps = len(T_room) # number of integration steps in the prediction horizon

        dev_neg = np.maximum(self.T_room_set_lower - np.reshape(T_room, (isteps, 1)), 0)
        dev_pos = np.maximum(np.reshape(T_room, (isteps, 1)) - self.T_room_set_upper, 0)

        return {
            'dev_neg_sum': float(np.sum(dev_neg) / isteps * timestep / 3600),
            'dev_neg_max': float(np.max(dev_neg)),
            'dev_pos_sum': float(np.sum(dev_pos) / isteps * timestep / 3600),
            'dev_pos_max': float(np.max(dev_pos))
        }

    def print_params(self):
        """ This function prints information about the selected building.
        """
        print(f"Building name: {self.params.get('name', 'unknown')}")
        print(f"Floor area: {self.params['area_floor']} m²")
        print(f"H_ve: {self.params['H_ve']:.2f} W/K")
        print(f"H_tr: {self.params['H_tr']:.2f} W/K")
        print(f"H_rad_con: {self.params['H_rad_con']:.2f} W/K")
        print(f"C_zone: {self.params['C_zone'] / 1000:.2f} kJ/K")
        print(f"C_wall: {self.params['C_wall'] / 1000:.2f} kJ/K")
        print(f"C_water: {self.params['C_water'] / 1000:.2f} kJ/K")

    def plot_step_response(self):
        """ This function plots the step response of the building envelope (excluding the thermal mass of the heating system) ,
        i. e. the thermal response of the room temperature to a step in ambient temperature from 0 to 20 degC.
        """
        import matplotlib.pyplot as plt

        # initialize heat pump and building model
        hp_model = model_hvac.Heatpump_AW(mdot_HP = 0.25)
        timestep = 3600 # sec
        sim = simulator.Model_simulator(bldg_model = self,
                                        hp_model   = hp_model,
                                        timestep   = timestep)

        # save original parameters
        C_water_original = self.params['C_water']
        H_rad_con_original = self.params['H_rad_con']
        H_con_floor_original = self.params['H_con_floor']
        H_tr_floor_original = self.params['H_tr_floor']

        # decouple building from heat supply system
        self.params['C_water'] = 0.00001 
        self.params['H_con_floor'] = 0.00001
        self.params['H_tr_floor'] = 0.00001
        self.params['H_rad_con'] = 0.00001

        timeconstant = self.params['C_bldg']/self.params['H_ve_tr'] # sec
        #print(timeconstant/3600)
        time = np.arange(0,10*timeconstant/timestep, 1)

        # initialize states dictionary
        state_keys = self.state_keys
        x_init = {key : 0 for key in state_keys}
        
        # initialize disturbance vector
        p = pd.DataFrame(index = time)
        p['T_amb'] = 20
        p['T_amb'][0] = 0
        p['Qdot_gains'], p['Qdot_int'], p['Qdot_sol'] = 0, 0, 0
        
        # simulate rule based
        results =  sim.simulate(x_init = x_init,p = p)
        T_room = results['states']['T_room']

        # plot step response
        fig, ax = plt.subplots()
        ax.plot(time, p['T_amb'], label = 'T_amb')
        ax.plot(time, T_room[0:len(time)-1], label = 'T_room')
        fig.legend(bbox_to_anchor=[0.5, 0.95], loc = 'center', ncol=2)
        ax.set_ylabel('Temperature [degC]')
        ax.set_xlabel('Time [h]')

        # reset parameters to original values
        self.params['C_water'] = C_water_original 
        self.params['H_rad_con'] = H_rad_con_original
        self.params['H_con_floor'] = H_con_floor_original
        self.params['H_tr_floor'] = H_tr_floor_original


if __name__ == "__main__":
    # load example building data
    from data import mfh_1958_1968

    # Initialize the building model
    building = mfh_1958_1968
    bldg_model = Building(params    = building, # More example buildings can be found in data/buildings/.
                          mdot_hp   = 0.27,          # Massflow of the heat supply system. [kg/s]
                          verbose   = True)