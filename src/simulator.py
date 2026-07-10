import numpy as np
from scipy.integrate import solve_ivp

# numpy >=2.0 renamed trapz -> trapezoid; keep working on both
_trapz = getattr(np, 'trapezoid', None) or np.trapz


class Simulator:
    """One-step building + heat pump simulator.

    Parameters
    ----------
    hp_model : iDM_AERO_ALM_4_12
        Heat pump model instance.
    bldg_model : Building
        Building model instance (4R3C).
    timestep : int
        Sampling interval [s]. Default 3600 (1 hour).
    """

    def __init__(self, hp_model, bldg_model, timestep=3600):
        self.hp_model   = hp_model
        self.bldg_model = bldg_model
        self.timestep   = timestep

    # ── One-step integration ──────────────────────────────────────────────────

    def get_next_state(self, x_init, uk, pk):
        """Advance the simulation by one timestep.

        Parameters
        ----------
        x_init : dict
            Current state: {'T_room': ..., 'T_wall': ..., 'T_hp_ret': ...}
        uk : float
            Control variable — HP supply temperature setpoint T_hp_sup [°C].
        pk : dict
            Disturbances: {'T_amb': ..., 'Qdot_gains': ...}

        Returns
        -------
        dict with keys:
            'state'      : dict  — next state
            'controls'   : float — applied uk
            'parameters' : dict  — pk
            'cost'       : dict  — energy + comfort costs
        """
        state_keys = self.bldg_model.state_keys
        input_keys  = self.bldg_model.input_keys

        # Build initial state array in the order the ODE expects
        x_init_np = np.array([x_init[key] for key in state_keys])

        # Real compressor capacity ceiling at this operating point [W].
        # This is what actually constrains how fast T_hp_ret can rise in
        # the ODE -- see Building.calc(). Without it, the simulator models
        # an idealized infinite-capacity heat source (see simulator notes).
        Qdot_hp_max_W = self.hp_model.get_max_heating_capacity(
            T_amb=pk['T_amb'], T_flow=uk) * 1000.0

        # Build input list: T_hp_sup first, then disturbances
        # Backup heater (Heizstab 6kW): fires automatically when HP is on
        # AND room is below the comfort lower bound (20°C). Rule-based,
        # not agent-controlled — matches real iDM controller behaviour.
        T_room_prev  = x_init['T_room']
        hp_is_on     = uk > 21.0   # HP running when supply setpoint above return+margin
        Q_backup_max = self.bldg_model.params.get('Q_backup_max', 0.0)
        Q_backup     = Q_backup_max if (hp_is_on and T_room_prev < 20.0) else 0.0
        input_dict   = {'T_hp_sup': uk, 'Qdot_hp_max': Qdot_hp_max_W,
                        'Q_backup': Q_backup, **pk}
        input_values = [input_dict[key] for key in input_keys]

        # Integrate ODE over one timestep
        ode_result = solve_ivp(
            fun    = self.bldg_model.calc,
            t_span = [0, self.timestep],
            y0     = x_init_np,
            args   = (input_values,),
            method = 'LSODA',
        )

        if not ode_result.success:
            print(f"[Simulator] ODE solver warning: {ode_result.message}")

        # ode_result.y shape: (n_states, n_time_points)
        # All intermediate states — used for cost calculation (like i4b)
        state_dict = {key: ode_result.y[i] for i, key in enumerate(state_keys)}

        # Final state only — becomes next x_init
        next_state = {key: float(ode_result.y[i, -1]) for i, key in enumerate(state_keys)}

        # Compute costs using intermediate states (not just final), passing
        # the solver's actual (non-uniform) time points for proper
        # time-weighted averaging instead of a flat per-sample mean.
        cost = self._calc_cost(
            state_dict=state_dict, input_dict=input_dict, t=ode_result.t)

        return {
            'state':      next_state,
            'controls':   uk,
            'parameters': pk,
            'cost':       cost,
        }

    # ── Cost calculation ──────────────────────────────────────────────────────

    def _calc_cost(self, state_dict, input_dict, t=None):
        """Compute HP energy cost and building comfort deviation.

        Uses all intermediate solve_ivp states (like i4b) for accuracy.

        Parameters
        ----------
        state_dict : dict
            Arrays of intermediate states from solve_ivp.
        input_dict : dict
            {'T_hp_sup': ..., 'T_amb': ..., 'Qdot_gains': ..., 'Qdot_hp_max': ...}
        t : array-like, optional
            The solver's actual (non-uniform) time points, for proper
            time-weighted (trapezoidal) averaging instead of a flat mean
            over solver steps -- see notes in calc_comfort_dev.

        Returns
        -------
        dict merging HP cost and comfort deviation keys.
        """
        T_hp_ret_trace = state_dict['T_hp_ret']
        T_room_trace   = state_dict['T_room']

        T_hp_sup    = input_dict['T_hp_sup']
        T_amb       = input_dict['T_amb']
        Qdot_hp_max = input_dict['Qdot_hp_max']
        Q_backup    = input_dict.get('Q_backup', 0.0)

        COP = self.hp_model.compute_COP(T_amb=T_amb, T_flow=T_hp_sup)

        Qdot_th_trace = np.clip(
            self.bldg_model.mdot_hp * 4181.0 * (T_hp_sup - T_hp_ret_trace),
            0.0, Qdot_hp_max,
        )

        if t is not None and len(t) > 1:
            Qdot_th = float(_trapz(Qdot_th_trace, t) / (t[-1] - t[0]))
        else:
            Qdot_th = float(np.mean(Qdot_th_trace))

        # ── Modulation floor ──────────────────────────────────────────────────
        Q_MOD_MIN = 4000.0
        if Qdot_th >= Q_MOD_MIN / 2.0:
            Qdot_th = max(Qdot_th, Q_MOD_MIN)
            hp_on   = True
        else:
            Qdot_th = 0.0
            hp_on   = False

        P_el_hp      = Qdot_th / COP if (COP > 0 and hp_on) else 0.0   # W
        # Backup heater is pure resistance: COP=1, P_el = Q_thermal
        # Only runs when HP is on (hp_on already accounts for modulation floor)
        P_el_backup  = Q_backup if hp_on else 0.0                       # W
        P_el         = P_el_hp + P_el_backup                            # W total
        E_el         = P_el * self.timestep / 3600.0                    # Wh

        cost_hp = {
            'COP':          float(COP),
            'Qdot_th':      float(Qdot_th),       # W  HP thermal only
            'Q_backup':     float(P_el_backup),    # W  backup thermal (=P_el_backup, COP=1)
            'P_el':         float(P_el),           # W  total electrical
            'P_el_hp':      float(P_el_hp),        # W  HP compressor only
            'P_el_backup':  float(P_el_backup),    # W  backup heater only
            'E_el':         float(E_el),           # Wh total
            'hp_on':        bool(hp_on),
            'backup_on':    bool(P_el_backup > 0),
        }

        # ── Comfort deviation ─────────────────────────────────────────────────
        cost_bldg = self.bldg_model.calc_comfort_dev(
            T_room   = T_room_trace,
            timestep = self.timestep,
            t        = t,
        )

        return {**cost_hp, **cost_bldg}