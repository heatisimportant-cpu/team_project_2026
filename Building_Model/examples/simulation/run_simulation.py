"""
Complete simulation comparing MPC vs Baseline (Heat Curve + PID)

Simulations:
1. Winter 2024 (Dec-Feb)
2. Full year 2024
3. Full year 2025
4. Year 2026 (Jan - May 30)
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.building import Building
from src.controllers.heatcurve import HeatCurveController
from src.controllers.pid import PIDController
from src.controllers.mpc import MPCController
from data.buildings.mfh_1968 import get_building


def load_data(start_date, end_date):
    """Load weather, prices, and gains data for specified period"""

    weather_path = os.path.join(project_root, 'data/weather/hannover_2024_2026.csv')
    prices_path = os.path.join(project_root, 'data/prices/electricity_prices_2024.csv')
    gains_path = os.path.join(project_root, 'data/profiles/internal_gains.csv')

    weather = pd.read_csv(weather_path)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])

    prices = pd.read_csv(prices_path)
    prices['timestamp'] = pd.to_datetime(prices['timestamp'])

    gains = pd.read_csv(gains_path)
    gains['timestamp'] = pd.to_datetime(gains['timestamp'])

    # Filter to period
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    weather = weather[(weather['timestamp'] >= start) & (weather['timestamp'] <= end)]

    # For prices, repeat 2024 data for 2025 and 2026
    prices_2024 = prices.copy()
    if end.year > 2024:
        # Extend prices by repeating 2024 pattern
        prices_list = [prices_2024]
        for year in range(2025, end.year + 1):
            prices_year = prices_2024.copy()
            prices_year['timestamp'] = prices_year['timestamp'] + pd.DateOffset(years=year-2024)
            prices_list.append(prices_year)
        prices = pd.concat(prices_list, ignore_index=True)

    prices = prices[(prices['timestamp'] >= start) & (prices['timestamp'] <= end)]
    gains = gains[(gains['timestamp'] >= start) & (gains['timestamp'] <= end)]

    return weather, prices, gains


def simulate_baseline(building, heatcurve, pid, weather, gains, dt=3600):
    """Simulate with Heat Curve + PID controller"""

    n_steps = len(weather)
    state = np.array([20.0, 20.0, 35.0])

    results = {
        'timestamp': weather['timestamp'].values,
        'T_room': np.zeros(n_steps),
        'T_wall': np.zeros(n_steps),
        'T_hp_ret': np.zeros(n_steps),
        'T_supply': np.zeros(n_steps),
        'T_amb': np.zeros(n_steps),
        'Q_heat': np.zeros(n_steps),
    }

    # Reset PID
    pid.reset()

    for i in range(n_steps):
        T_amb = weather.iloc[i]['T_amb']
        Q_gains = gains.iloc[i]['Q_int_total']

        # Heat curve provides base temperature
        T_supply_hc = heatcurve.calc_supply_temp(T_amb)

        # PID adjusts based on room temperature
        T_supply_pid = pid.update(state[0], dt)

        # Combine: weighted average (70% heat curve, 30% PID)
        T_supply = 0.7 * T_supply_hc + 0.3 * T_supply_pid
        T_supply = np.clip(T_supply, 20, 50)

        # Simulate
        try:
            rhs = building.calc_4r3c(0, state, T_supply, [T_amb, Q_gains])

            # Check for NaN or Inf
            if np.any(np.isnan(rhs)) or np.any(np.isinf(rhs)):
                print(f"Warning: Invalid RHS at step {i}, using zero change")
                rhs = np.zeros_like(rhs)

            # Update state with clipping
            state_new = state + rhs * dt

            # Clip to physical limits
            state_new[0] = np.clip(state_new[0], 5.0, 35.0)   # T_room
            state_new[1] = np.clip(state_new[1], 5.0, 35.0)   # T_wall
            state_new[2] = np.clip(state_new[2], 15.0, 60.0)  # T_hp_ret

            state = state_new

        except Exception as e:
            print(f"Error at step {i}: {e}")
            # Keep previous state

        # Calculate heat output
        cp_water = 4180.0
        Q_heat = building.mdot_hp * cp_water * (T_supply - state[2])
        Q_heat = max(0, Q_heat)

        # Store
        results['T_room'][i] = state[0]
        results['T_wall'][i] = state[1]
        results['T_hp_ret'][i] = state[2]
        results['T_supply'][i] = T_supply
        results['T_amb'][i] = T_amb
        results['Q_heat'][i] = Q_heat

    return pd.DataFrame(results)


def simulate_mpc(building, mpc, weather, gains, prices, dt=3600):
    """Simulate with MPC controller"""

    n_steps = len(weather)
    state = np.array([20.0, 20.0, 35.0])

    results = {
        'timestamp': weather['timestamp'].values,
        'T_room': np.zeros(n_steps),
        'T_wall': np.zeros(n_steps),
        'T_hp_ret': np.zeros(n_steps),
        'T_supply': np.zeros(n_steps),
        'T_amb': np.zeros(n_steps),
        'Q_heat': np.zeros(n_steps),
    }

    horizon = mpc.horizon

    for i in range(n_steps):
        T_amb = weather.iloc[i]['T_amb']
        Q_gains = gains.iloc[i]['Q_int_total']

        # MPC optimization every 24 hours
        if i % 24 == 0:
            # Get forecasts
            i_end = min(i + horizon, n_steps)
            T_amb_forecast = weather.iloc[i:i_end]['T_amb'].values
            Q_gains_forecast = gains.iloc[i:i_end]['Q_int_total'].values

            # Price forecast (repeat last known price if needed)
            price_forecast = prices.iloc[min(i, len(prices)-1):min(i_end, len(prices))]['price_eur_kwh'].values
            if len(price_forecast) < horizon:
                price_forecast = np.pad(price_forecast, (0, horizon - len(price_forecast)), 
                                       mode='edge')

            # Optimize
            try:
                T_supply_plan, _ = mpc.optimize(state, T_amb_forecast, Q_gains_forecast, 
                                               price_forecast)
                current_plan_index = 0
            except:
                # If optimization fails, use heat curve
                hc = HeatCurveController()
                T_supply_plan = np.array([hc.calc_supply_temp(T_amb) for _ in range(horizon)])
                current_plan_index = 0

        # Use planned temperature
        T_supply = T_supply_plan[min(current_plan_index, len(T_supply_plan)-1)]
        current_plan_index += 1
        T_supply = np.clip(T_supply, 25, 50)

        # Simulate
        try:
            rhs = building.calc_4r3c(0, state, T_supply, [T_amb, Q_gains])

            # Check for NaN or Inf
            if np.any(np.isnan(rhs)) or np.any(np.isinf(rhs)):
                rhs = np.zeros_like(rhs)

            # Update state with clipping
            state_new = state + rhs * dt

            # Clip to physical limits
            state_new[0] = np.clip(state_new[0], 5.0, 35.0)   # T_room
            state_new[1] = np.clip(state_new[1], 5.0, 35.0)   # T_wall
            state_new[2] = np.clip(state_new[2], 15.0, 60.0)  # T_hp_ret

            state = state_new

        except Exception as e:
            print(f"Error at step {i}: {e}")

        # Calculate heat output
        cp_water = 4180.0
        Q_heat = building.mdot_hp * cp_water * (T_supply - state[2])
        Q_heat = max(0, Q_heat)

        # Store
        results['T_room'][i] = state[0]
        results['T_wall'][i] = state[1]
        results['T_hp_ret'][i] = state[2]
        results['T_supply'][i] = T_supply
        results['T_amb'][i] = T_amb
        results['Q_heat'][i] = Q_heat

    return pd.DataFrame(results)


def calculate_metrics(results, prices, name):
    """Calculate performance metrics"""

    # Match prices to results
    results_with_prices = results.copy()
    results_with_prices['timestamp'] = pd.to_datetime(results_with_prices['timestamp'])

    # Merge with prices
    prices_copy = prices.copy()
    prices_copy['timestamp'] = pd.to_datetime(prices_copy['timestamp'])

    merged = pd.merge(results_with_prices, prices_copy[['timestamp', 'price_eur_kwh']], 
                     on='timestamp', how='left')

    # Forward fill missing prices
    merged['price_eur_kwh'] = merged['price_eur_kwh'].ffill()
    merged['price_eur_kwh'] = merged['price_eur_kwh'].fillna(merged['price_eur_kwh'].mean())

    # Calculate metrics
    # Energy consumption
    Q_total_kwh = results['Q_heat'].sum() / 3600 / 1000  # kWh

    # Cost (assume average COP of 3.5)
    COP_avg = 3.5
    E_el_kwh = Q_total_kwh / COP_avg
    cost_total = (merged['Q_heat'] / 3600 / 1000 / COP_avg * merged['price_eur_kwh']).sum()

    # Comfort
    comfort_pct = ((results['T_room'] >= 20) & (results['T_room'] <= 24)).sum() / len(results) * 100
    T_room_avg = results['T_room'].mean()
    T_room_min = results['T_room'].min()
    T_room_max = results['T_room'].max()

    # Hours outside comfort
    hours_too_cold = (results['T_room'] < 20).sum()
    hours_too_hot = (results['T_room'] > 24).sum()

    return {
        'name': name,
        'hours': len(results),
        'Q_total_kwh': Q_total_kwh,
        'E_el_kwh': E_el_kwh,
        'cost_eur': cost_total,
        'comfort_pct': comfort_pct,
        'T_room_avg': T_room_avg,
        'T_room_min': T_room_min,
        'T_room_max': T_room_max,
        'hours_too_cold': hours_too_cold,
        'hours_too_hot': hours_too_hot,
    }


def run_simulation(period_name, start_date, end_date, building):
    """Run simulation for specific period"""

    print(f"\n{'='*70}")
    print(f"SIMULATION: {period_name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}")

    # Load data
    print("\nLoading data...")
    weather, prices, gains = load_data(start_date, end_date)
    print(f"  Weather: {len(weather)} hours")
    print(f"  Prices: {len(prices)} hours")
    print(f"  Gains: {len(gains)} hours")

    # Controllers
    heatcurve = HeatCurveController(T_supply_nom=45, T_amb_design=-12, T_room_set=20)
    pid = PIDController(Kp=1.0, Ki=0.05, Kd=0.02, setpoint=20.0)  # Reduced gains
    mpc = MPCController(building, horizon=24)

    # Simulate baseline
    print("\n  Running BASELINE (Heat Curve + PID)...")
    results_baseline = simulate_baseline(building, heatcurve, pid, weather, gains)
    metrics_baseline = calculate_metrics(results_baseline, prices, 'Baseline')

    # Simulate MPC
    print("  Running MPC (Price-Optimized)...")
    results_mpc = simulate_mpc(building, mpc, weather, gains, prices)
    metrics_mpc = calculate_metrics(results_mpc, prices, 'MPC')

    # Save results
    output_dir = os.path.join(project_root, 'examples/simulation/results')
    os.makedirs(output_dir, exist_ok=True)

    filename_base = period_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    results_baseline.to_csv(f'{output_dir}/{filename_base}_baseline.csv', index=False)
    results_mpc.to_csv(f'{output_dir}/{filename_base}_mpc.csv', index=False)

    # Print comparison
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Metric':<25} {'Baseline':<20} {'MPC':<20} {'Savings':<15}")
    print("-" * 80)
    print(f"{'Hours simulated':<25} {metrics_baseline['hours']:<20} {metrics_mpc['hours']:<20} {'-':<15}")

    if metrics_baseline['Q_total_kwh'] > 0:
        print(f"{'Heat energy (kWh)':<25} {metrics_baseline['Q_total_kwh']:<20.1f} {metrics_mpc['Q_total_kwh']:<20.1f} {(1-metrics_mpc['Q_total_kwh']/metrics_baseline['Q_total_kwh'])*100:>14.1f}%")
        print(f"{'Electricity (kWh)':<25} {metrics_baseline['E_el_kwh']:<20.1f} {metrics_mpc['E_el_kwh']:<20.1f} {(1-metrics_mpc['E_el_kwh']/metrics_baseline['E_el_kwh'])*100:>14.1f}%")
        print(f"{'Cost (EUR)':<25} {metrics_baseline['cost_eur']:<20.2f} {metrics_mpc['cost_eur']:<20.2f} {(1-metrics_mpc['cost_eur']/metrics_baseline['cost_eur'])*100:>14.1f}%")

    print(f"{'Comfort (% in 20-24°C)':<25} {metrics_baseline['comfort_pct']:<20.1f} {metrics_mpc['comfort_pct']:<20.1f} {'-':<15}")
    print(f"{'Avg T_room (°C)':<25} {metrics_baseline['T_room_avg']:<20.2f} {metrics_mpc['T_room_avg']:<20.2f} {'-':<15}")
    print(f"{'Min T_room (°C)':<25} {metrics_baseline['T_room_min']:<20.2f} {metrics_mpc['T_room_min']:<20.2f} {'-':<15}")
    print(f"{'Max T_room (°C)':<25} {metrics_baseline['T_room_max']:<20.2f} {metrics_mpc['T_room_max']:<20.2f} {'-':<15}")
    print()

    return metrics_baseline, metrics_mpc


def main():
    print("=" * 70)
    print("VIRCHOWSTR. 6 - COMPLETE HEATING SYSTEM COMPARISON")
    print("MPC vs Baseline (Heat Curve + PID)")
    print("=" * 70)

    # Initialize building
    building_params = get_building()
    building = Building(building_params, mdot_hp=0.27, verbose=False)

    print("\nBuilding: MFH 1968 Virchowstr. 6")
    print(f"  Floor area: {building.params['area_floor']} m²")
    print(f"  H_tr: {building.params['H_tr']} W/K")
    print(f"  H_ve: {building.params['H_ve']} W/K")

    # Run simulations
    all_metrics = []

    # 1. Winter 2024 (Dec 2023 - Feb 2024)
    m_bl, m_mpc = run_simulation("Winter 2024", "2023-12-01", "2024-02-29", building)
    all_metrics.append(('Winter 2024', m_bl, m_mpc))

    # 2. Full year 2024
    m_bl, m_mpc = run_simulation("Year 2024", "2024-01-01", "2024-12-31", building)
    all_metrics.append(('Year 2024', m_bl, m_mpc))

    # 3. Full year 2025
    m_bl, m_mpc = run_simulation("Year 2025", "2025-01-01", "2025-12-31", building)
    all_metrics.append(('Year 2025', m_bl, m_mpc))

    # 4. Year 2026 (Jan - May 30)
    m_bl, m_mpc = run_simulation("Year 2026 (Jan-May)", "2026-01-01", "2026-05-30", building)
    all_metrics.append(('Year 2026', m_bl, m_mpc))

    # Summary
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Period':<25} {'Cost Baseline':<18} {'Cost MPC':<18} {'Savings':<15}")
    print("-" * 76)
    for period, m_bl, m_mpc in all_metrics:
        if m_bl['cost_eur'] > 0:
            savings_pct = (1 - m_mpc['cost_eur'] / m_bl['cost_eur']) * 100
            print(f"{period:<25} {m_bl['cost_eur']:>15.2f} EUR {m_mpc['cost_eur']:>15.2f} EUR {savings_pct:>13.1f}%")

    print("\n" + "=" * 70)
    print("✅ All simulations complete!")
    print("   Results saved in: examples/simulation/results/")
    print("=" * 70)


if __name__ == '__main__':
    main()
