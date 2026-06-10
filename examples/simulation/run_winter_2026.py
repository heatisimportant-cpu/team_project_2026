"""
Winter 2026 Heating Comparison with Thermal Storage Tank
MPC uses tank to store heat during cheap prices and maintain 21°C room temp

Solar gains are read from data/profiles/internal_gains.csv (Q_sol column),
computed from DWD measured irradiance data for Hannover.
Re-run data/solar/calc_solar_gains.py to refresh the solar gains profile.
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from data.buildings.mfh_1968 import get_building

from src.config import ACTIVE_BUILDING
building = ACTIVE_BUILDING


def deadband_error(T_room, T_set, band=1.0):
    """
    Deadband around setpoint: returns 0 inside [T_set-band, T_set+band],
    positive when too cold, negative when too hot.
    """
    low = T_set - band
    high = T_set + band
    if T_room < low:
        return low - T_room
    elif T_room > high:
        return high - T_room
    else:
        return 0.0



def load_winter_data():
    """Load winter 2026 data"""

    weather = pd.read_csv(f'{project_root}/data/weather/hannover_2024_2026.csv')
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])

    prices = pd.read_csv(f'{project_root}/data/prices/electricity_prices_2024.csv')
    prices['timestamp'] = pd.to_datetime(prices['timestamp'])

    gains = pd.read_csv(f'{project_root}/data/profiles/internal_gains.csv')
    gains['timestamp'] = pd.to_datetime(gains['timestamp'])

    # Winter period: Dec 1, 2025 - Feb 28, 2026
    start = pd.to_datetime('2025-12-01')
    end = pd.to_datetime('2026-02-28')

    weather = weather[(weather['timestamp'] >= start) & (weather['timestamp'] <= end)].reset_index(drop=True)
    gains = gains[(gains['timestamp'] >= start) & (gains['timestamp'] <= end)].reset_index(drop=True)

    # Extend prices to 2025/2026
    prices_2025 = prices.copy()
    prices_2025['timestamp'] = prices_2025['timestamp'] + pd.DateOffset(years=1)
    prices_all = pd.concat([prices, prices_2025], ignore_index=True)
    prices_all = prices_all[(prices_all['timestamp'] >= start) & (prices_all['timestamp'] <= end)].reset_index(drop=True)

    return weather, prices_all, gains


def simulate_baseline(weather, gains, building):
    """Baseline: Heating curve with 21°C setpoint, no storage"""

    results = []
    T_room = 21.0
    T_setpoint = 21.0

    for i in range(len(weather)):
        T_amb   = weather.iloc[i]['T_amb']
        Q_gains = gains.iloc[i]['Q_int_total']          # internal + solar gains
        Q_sol   = gains.iloc[i]['Q_sol'] if 'Q_sol' in gains.columns else 0.0

        error = deadband_error(T_room, T_setpoint, band=1.0)

        T_supply_base = 25 + 0.9 * (T_setpoint - T_amb)
        T_supply = T_supply_base + 10 * error
        T_supply = np.clip(T_supply, 25, 50)

        H_tot = building['H_tr'] + building['H_ve']
        Q_loss = H_tot * (T_room - T_amb)

        Q_heat = building['mdot_hp'] * 4180 * (T_supply - 35)
        Q_heat = max(0, Q_heat)

        C_room = building['c_bldg'] * building['area_floor'] * 3600
        dT = (Q_heat + Q_gains - Q_loss) / C_room * 3600

        T_room += np.clip(dT, -1.5, 1.5)
        T_room = np.clip(T_room, 20.0, 22.0)

        results.append({
            'timestamp': weather.iloc[i]['timestamp'],
            'T_room': T_room,
            'T_supply': T_supply,
            'T_amb': T_amb,
            'Q_heat': Q_heat,
            'Q_gains': Q_gains,
            'Q_sol': Q_sol,
            'SOC': 0.0  # No tank in baseline
        })

    return pd.DataFrame(results)


def simulate_mpc_with_tank(weather, gains, prices, building):
    """MPC: Uses thermal storage tank to shift load to cheap hours, maintains 21°C"""

    results = []
    T_room = 21.0
    T_setpoint = 21.0

    # Tank parameters (500L buffer tank)
    tank_volume = 500  # liters
    tank_capacity = tank_volume * 4.18 * 40  # kJ (can store 40°C delta)
    tank_capacity_kwh = tank_capacity / 3600  # kWh
    SOC = 0.5  # Start at 50% state of charge
    SOC_min = 0.1
    SOC_max = 0.95
    tank_loss_rate = 0.02  # 2% loss per hour

    # Compute price percentiles
    price_vals = prices['price_eur_kwh'].values
    price_low = np.percentile(price_vals, 25)
    price_high = np.percentile(price_vals, 75)

    for i in range(len(weather)):
        T_amb    = weather.iloc[i]['T_amb']
        Q_gains  = gains.iloc[i]['Q_int_total']         # internal + solar gains
        Q_sol    = gains.iloc[i]['Q_sol'] if 'Q_sol' in gains.columns else 0.0
        price_now = prices.iloc[i]['price_eur_kwh'] if i < len(prices) else 0.30

        # Calculate heating demand
        H_tot = building['H_tr'] + building['H_ve']
        Q_loss = H_tot * (T_room - T_amb)
        error = deadband_error(T_room, T_setpoint, band=1.0)

        # MPC decision: charge tank during cheap hours, use tank during expensive hours
        charge_tank = False
        use_tank = False

        # Look ahead at next 6 hours
        if i < len(prices) - 6:
            next_6h_prices = prices.iloc[i:i+6]['price_eur_kwh'].mean()

            # If current price is low and next hours are higher, charge tank
            if price_now < price_low and SOC < SOC_max and next_6h_prices > price_now * 1.1:
                charge_tank = True

            # If current price is high and tank has energy, use tank
            if price_now > price_high and SOC > SOC_min:
                use_tank = True

        # Heat pump operation
        if use_tank and SOC > SOC_min:
            # Use stored heat from tank, minimal HP operation
            T_supply_base = 25 + 0.7 * (T_setpoint - T_amb)
            T_supply = T_supply_base + 8 * error

            # Take heat from tank
            Q_from_tank = min(SOC * tank_capacity / 3600, 3000)  # Max 3 kW from tank
            SOC -= Q_from_tank * 3600 / tank_capacity
            SOC = max(SOC, SOC_min)

        elif charge_tank and SOC < SOC_max:
            # Charge tank during cheap hours
            T_supply_base = 25 + 1.0 * (T_setpoint - T_amb) 
            T_supply = T_supply_base + 10 * error + 5  # Extra heating for tank
            Q_from_tank = 0

        else:
            # Normal operation
            T_supply_base = 25 + 0.85 * (T_setpoint - T_amb)
            T_supply = T_supply_base + 9 * error
            Q_from_tank = 0

        T_supply = np.clip(T_supply, 25, 50)

        # Heat pump output
        Q_hp = building['mdot_hp'] * 4180 * (T_supply - 35)
        Q_hp = max(0, Q_hp)

        # Heat delivered to room
        Q_heat = Q_hp + Q_from_tank

        # Charge tank if excess heat available
        Q_needed = Q_loss - Q_gains
        Q_excess = Q_hp - Q_needed

        if Q_excess > 500 and SOC < SOC_max and charge_tank:
            Q_to_tank = min(Q_excess * 0.7, 4000)  # Store up to 4 kW
            SOC += Q_to_tank * 3600 / tank_capacity
            SOC = min(SOC, SOC_max)
            Q_heat = Q_hp - Q_to_tank + Q_from_tank

        # Tank losses
        SOC *= (1 - tank_loss_rate)
        SOC = np.clip(SOC, 0, 1)

        # Room temperature update
        C_room = building['c_bldg'] * building['area_floor'] * 3600
        dT = (Q_heat + Q_gains - Q_loss) / C_room * 3600

        T_room += np.clip(dT, -1.5, 1.5)
        T_room = np.clip(T_room, 20.0, 22.0)

        results.append({
            'timestamp': weather.iloc[i]['timestamp'],
            'T_room': T_room,
            'T_supply': T_supply,
            'T_amb': T_amb,
            'Q_heat': Q_heat,
            'Q_gains': Q_gains,
            'Q_sol': Q_sol,
            'SOC': SOC * 100  # Store as percentage
        })

    return pd.DataFrame(results)


def calculate_cop(T_supply, T_amb):
    """Calculate heat pump COP"""
    T_hot = T_supply + 273.15
    T_cold = T_amb + 273.15
    COP_carnot = T_hot / max(T_hot - T_cold, 1)
    COP = 0.45 * COP_carnot
    return np.clip(COP, 2.0, 6.0)


def analyze_results(results, prices, name):
    """Calculate performance metrics"""

    merged = pd.merge(results, prices[['timestamp', 'price_eur_kwh']], 
                     on='timestamp', how='left')
    merged['price_eur_kwh'] = merged['price_eur_kwh'].fillna(0.30)

    COPs = [calculate_cop(row['T_supply'], row['T_amb']) 
            for _, row in results.iterrows()]

    Q_total = results['Q_heat'].sum() / 1000
    E_el = sum(results['Q_heat'].iloc[i] / COPs[i] / 1000 
               for i in range(len(results)))
    cost = sum(results['Q_heat'].iloc[i] / COPs[i] / 1000 * merged['price_eur_kwh'].iloc[i]
               for i in range(len(results)))

    T_room = results['T_room'].values
    comfort_20_22 = ((T_room >= 20) & (T_room <= 22)).sum() / len(T_room) * 100
    comfort_20_24 = ((T_room >= 20) & (T_room <= 24)).sum() / len(T_room) * 100

    return {
        'name': name,
        'hours': len(results),
        'heat_kwh': Q_total,
        'elec_kwh': E_el,
        'cost_eur': cost,
        'cop_avg': np.mean(COPs),
        'comfort_20_22': comfort_20_22,
        'comfort_20_24': comfort_20_24,
        'T_room_avg': T_room.mean(),
        'T_room_min': T_room.min(),
        'T_room_max': T_room.max(),
        'T_supply_avg': results['T_supply'].mean(),
        'hours_cold': (T_room < 20).sum(),
        'hours_hot': (T_room > 22).sum(),
        'soc_avg': results['SOC'].mean() if 'SOC' in results else 0
    }


def create_plots(df_bl, df_mpc, output_dir):
    """Create comparison plots - multiple 15-day intervals"""

    total_hours = len(df_bl)
    interval_hours = 360

    num_intervals = (total_hours + interval_hours - 1) // interval_hours

    print(f"Creating {num_intervals} graphs (15-day intervals)...")

    for interval_idx in range(num_intervals):
        start_hour = interval_idx * interval_hours
        end_hour = min(start_hour + interval_hours, total_hours)
        n = end_hour - start_hour

        if n < 24:
            continue

        hours = np.arange(n)

        fig, axes = plt.subplots(7, 1, figsize=(16, 18))

        df_bl_slice = df_bl.iloc[start_hour:end_hour]
        df_mpc_slice = df_mpc.iloc[start_hour:end_hour]

        # 1. Room Temperature
        axes[0].plot(hours, df_bl_slice['T_room'].values, label='Baseline', linewidth=1.8, color='#2E86AB')
        axes[0].plot(hours, df_mpc_slice['T_room'].values, label='MPC + Tank', linewidth=1.8, color='#A23B72')
        axes[0].axhline(21, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Target 21°C')
        axes[0].fill_between(hours, 20, 22, alpha=0.1, color='green')
        axes[0].set_ylabel('Room Temp [°C]', fontsize=10)
        axes[0].set_ylim([19, 23])
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.2)

        start_date = df_bl_slice.iloc[0]['timestamp'].strftime('%b %d')
        end_date = df_bl_slice.iloc[-1]['timestamp'].strftime('%b %d, %Y')
        axes[0].set_title(f'Winter 2026 with Thermal Storage - Days {interval_idx*15+1}-{interval_idx*15+15} ({start_date} - {end_date})', 
                         fontsize=12, fontweight='bold')

        # 2. Supply Temperature
        axes[1].plot(hours, df_bl_slice['T_supply'].values, label='Baseline', linewidth=1.8, color='#2E86AB')
        axes[1].plot(hours, df_mpc_slice['T_supply'].values, label='MPC + Tank', linewidth=1.8, color='#A23B72')
        axes[1].set_ylabel('Supply Temp [°C]', fontsize=10)
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(True, alpha=0.2)

        # 3. Heat Output
        axes[2].plot(hours, df_bl_slice['Q_heat'].values/1000, label='Baseline', linewidth=1.8, color='#2E86AB')
        axes[2].plot(hours, df_mpc_slice['Q_heat'].values/1000, label='MPC + Tank', linewidth=1.8, color='#A23B72')
        axes[2].set_ylabel('Heat Output [kW]', fontsize=10)
        axes[2].legend(loc='best', fontsize=9)
        axes[2].grid(True, alpha=0.2)

        # 4. COP
        cop_bl = [calculate_cop(df_bl_slice['T_supply'].iloc[i], df_bl_slice['T_amb'].iloc[i]) for i in range(n)]
        cop_mpc = [calculate_cop(df_mpc_slice['T_supply'].iloc[i], df_mpc_slice['T_amb'].iloc[i]) for i in range(n)]
        axes[3].plot(hours, cop_bl, label='Baseline COP', linewidth=1.8, color='#2E86AB')
        axes[3].plot(hours, cop_mpc, label='MPC COP', linewidth=1.8, color='#A23B72')
        axes[3].set_ylabel('COP [-]', fontsize=10)
        axes[3].legend(loc='best', fontsize=9)
        axes[3].grid(True, alpha=0.2)

        # 5. Tank State of Charge (SOC)
        axes[4].plot(hours, df_mpc_slice['SOC'].values, linewidth=2.0, color='#F18F01', label='Tank SOC')
        axes[4].axhline(50, color='gray', linestyle='--', alpha=0.4)
        axes[4].fill_between(hours, 0, 100, alpha=0.05, color='orange')
        axes[4].set_ylabel('Tank SOC [%]', fontsize=10)
        axes[4].set_ylim([0, 100])
        axes[4].legend(loc='best', fontsize=9)
        axes[4].grid(True, alpha=0.2)

        # 6. Solar Gains
        if 'Q_sol' in df_bl_slice.columns:
            axes[5].fill_between(hours, df_bl_slice['Q_sol'].values / 1000,
                                 alpha=0.6, color='#FFD700', label='Solar gains')
            axes[5].plot(hours, df_bl_slice['Q_sol'].values / 1000,
                         linewidth=1.2, color='#DAA520')
        axes[5].set_ylabel('Solar Gains [kW]', fontsize=10)
        axes[5].legend(loc='best', fontsize=9)
        axes[5].grid(True, alpha=0.2)

        # 7. Ambient Temperature
        axes[6].plot(hours, df_bl_slice['T_amb'].values, linewidth=1.8, color='#06A77D', label='Outdoor')
        axes[6].set_ylabel('Ambient Temp [°C]', fontsize=10)
        axes[6].set_xlabel('Hour', fontsize=10)
        axes[6].legend(loc='best', fontsize=9)
        axes[6].grid(True, alpha=0.2)

        plt.tight_layout()
        filename = f'{output_dir}/winter_2026_days_{interval_idx*15+1:03d}_{interval_idx*15+15:03d}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"  Saved: winter_2026_days_{interval_idx*15+1:03d}_{interval_idx*15+15:03d}.png")

    print(f"All {num_intervals} graphs created")


def main():
    print("\n" + "="*70)
    print("Winter 2026 Heating System Comparison with Thermal Storage")
    print("MFH 1968 Virchowstr. 6, Langenhagen")
    print("Target room temperature: 21°C")
    print("="*70)

    building = get_building('enev')
    print(f"\nBuilding: {building['name']}")
    print(f"Floor area: {building['area_floor']} m²")
    print(f"H_tr: {building['H_tr']} W/K")
    print(f"H_ve: {building['H_ve']} W/K")

    print("\nThermal Storage Tank: 500L buffer tank")
    print("Strategy: Charge during cheap hours, use during expensive hours")

    print("\nLoading weather, prices, and gains data...")
    weather, prices, gains = load_winter_data()
    print(f"Period: Dec 1, 2025 - Feb 28, 2026")
    print(f"Total hours: {len(weather)}")

    print("\nSimulating...")
    print("  [1/2] Baseline (heating curve, no storage)...")
    results_bl = simulate_baseline(weather, gains, building)

    print("  [2/2] MPC with thermal storage tank...")
    results_mpc = simulate_mpc_with_tank(weather, gains, prices, building)

    output_dir = f'{project_root}/examples/simulation/results'
    os.makedirs(output_dir, exist_ok=True)

    results_bl.to_csv(f'{output_dir}/winter_2026_baseline.csv', index=False)
    results_mpc.to_csv(f'{output_dir}/winter_2026_mpc.csv', index=False)
    print(f"\nCSV files saved (full {len(results_bl)} hours)")

    print("\nCreating 15-day interval graphs...")
    create_plots(results_bl, results_mpc, output_dir)

    metrics_bl = analyze_results(results_bl, prices, 'Baseline')
    metrics_mpc = analyze_results(results_mpc, prices, 'MPC + Tank')

    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"\n{'Metric':<30} {'Baseline':>15} {'MPC+Tank':>15} {'Savings':>12}")
    print("-"*72)

    print(f"{'Hours simulated':<30} {metrics_bl['hours']:>15} {metrics_mpc['hours']:>15} {'-':>12}")
    print(f"{'Heat energy (kWh)':<30} {metrics_bl['heat_kwh']:>15.1f} {metrics_mpc['heat_kwh']:>15.1f} {100*(metrics_bl['heat_kwh']-metrics_mpc['heat_kwh'])/metrics_bl['heat_kwh']:>11.1f}%")
    print(f"{'Electricity (kWh)':<30} {metrics_bl['elec_kwh']:>15.1f} {metrics_mpc['elec_kwh']:>15.1f} {100*(metrics_bl['elec_kwh']-metrics_mpc['elec_kwh'])/metrics_bl['elec_kwh']:>11.1f}%")
    print(f"{'Cost (EUR)':<30} {metrics_bl['cost_eur']:>15.2f} {metrics_mpc['cost_eur']:>15.2f} {100*(metrics_bl['cost_eur']-metrics_mpc['cost_eur'])/metrics_bl['cost_eur']:>11.1f}%")

    # Solar gain summary (same for both controllers – it's an input)
    q_sol_kwh = results_bl['Q_sol'].sum() / 1000.0
    q_sol_avg = results_bl['Q_sol'].mean()
    print(f"{'Solar gains (kWh)':<30} {q_sol_kwh:>15.1f} {'(same)':>15} {'-':>12}")
    print(f"{'Solar gains avg (W)':<30} {q_sol_avg:>15.1f} {'(same)':>15} {'-':>12}")
    print(f"{'Average COP':<30} {metrics_bl['cop_avg']:>15.2f} {metrics_mpc['cop_avg']:>15.2f} {'-':>12}")
    print(f"{'Avg supply temp (°C)':<30} {metrics_bl['T_supply_avg']:>15.1f} {metrics_mpc['T_supply_avg']:>15.1f} {'-':>12}")
    print(f"{'Avg tank SOC (%)':<30} {metrics_bl['soc_avg']:>15.1f} {metrics_mpc['soc_avg']:>15.1f} {'-':>12}")
    print(f"{'Comfort 20-22°C (%)':<30} {metrics_bl['comfort_20_22']:>15.1f} {metrics_mpc['comfort_20_22']:>15.1f} {'-':>12}")
    print(f"{'Comfort 20-24°C (%)':<30} {metrics_bl['comfort_20_24']:>15.1f} {metrics_mpc['comfort_20_24']:>15.1f} {'-':>12}")
    print(f"{'Avg room temp (°C)':<30} {metrics_bl['T_room_avg']:>15.2f} {metrics_mpc['T_room_avg']:>15.2f} {'-':>12}")
    print(f"{'Min room temp (°C)':<30} {metrics_bl['T_room_min']:>15.2f} {metrics_mpc['T_room_min']:>15.2f} {'-':>12}")
    print(f"{'Max room temp (°C)':<30} {metrics_bl['T_room_max']:>15.2f} {metrics_mpc['T_room_max']:>15.2f} {'-':>12}")
    print(f"{'Hours too cold (<20°C)':<30} {metrics_bl['hours_cold']:>15} {metrics_mpc['hours_cold']:>15} {'-':>12}")
    print(f"{'Hours too hot (>22°C)':<30} {metrics_bl['hours_hot']:>15} {metrics_mpc['hours_hot']:>15} {'-':>12}")

    print("\n" + "="*70)
    print("Simulation complete")
    print("="*70)
    print(f"\nOutput files in: {output_dir}")
    print(f"  CSV (full data):")
    print(f"    - winter_2026_baseline.csv")
    print(f"    - winter_2026_mpc.csv")
    print(f"  Graphs (15-day intervals with tank SOC):")
    print(f"    - winter_2026_days_001_015.png")
    print(f"    - winter_2026_days_016_030.png")
    print(f"    - winter_2026_days_031_045.png")
    print(f"    - winter_2026_days_046_060.png")
    print(f"    - winter_2026_days_061_075.png")
    print(f"    - winter_2026_days_076_090.png")
    print()


if __name__ == '__main__':
    main()
