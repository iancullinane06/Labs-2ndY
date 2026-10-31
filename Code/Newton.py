from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import linregress
import seaborn as sns
import numpy as np
import tools.Styling as Styling
from tools.Symbols import symbols

# Measured quantities and their uncertainties (SI units)
calorimeter_mass = 0.003489
calorimeter_mass_uncertainty = 0.0000005
water_mass = 0.150061
water_mass_uncertainty = 0.000005
surface_area = 0.0145947
surface_area_uncertainty = 0.00001
voltage = 2.98
voltage_uncertainty = 0.005
heating_current = 1.25
heating_current_uncertainty = 0.005

# Model constants
air_thermal_conductivity = 0.026
aluminium_specific_heat_capacity = 900
water_specific_heat_capacity = 4180

calorimeter_surface_area = 0.0145947 
calorimeter_surface_area_uncertainty = 0.00001
heat_capacity = (calorimeter_mass * aluminium_specific_heat_capacity + water_mass * water_specific_heat_capacity)
heat_capacity_uncertainty = np.sqrt(
    (aluminium_specific_heat_capacity * calorimeter_mass_uncertainty) ** 2
    + (water_specific_heat_capacity * water_mass_uncertainty) ** 2
)

# Adiabatic-index apparatus measurements (SI units)
ball_mass = 0.01482
ball_mass_uncertainty = 0.0000005
ball_cross_sectional_area = 0.000201
ball_cross_sectional_area_uncertainty = 0.000001
enclosed_air_volume = 0.01
enclosed_air_volume_uncertainty = 0.0
equilibrium_pressure = 100342.2
equilibrium_pressure_uncertainty = 50

data = pd.read_csv("Code/Data/Newton.csv")
data_drops = pd.read_csv("Code/Data/Newton_drop.csv")
data_drops.columns = data_drops.columns.str.strip()

figure_size = (10, 6)
report_images_dir = "Reports/Report 6 - Newton/images"

def analyse_heating(data):
    delta_T_heating = -(data["Heating Temp"] - data["Heating Temp bg"]).to_numpy(dtype=float)
    time_seconds = data["Heating Time"].to_numpy(dtype=float) * 60
    valid = np.isfinite(time_seconds) & np.isfinite(delta_T_heating) & (delta_T_heating > 0)
    if valid.sum() < 2:
        raise ValueError("At least two heating measurements above ambient are required.")

    fit = linregress(time_seconds[valid], (delta_T_heating[valid]))
    heating_coefficient = fit.slope
    heating_coefficient_uncertainty = fit.stderr
    print(
        f"Heating rate = {heating_coefficient:.3e} +/- "
        f"{heating_coefficient_uncertainty:.1e} K s^-1 "
        f"(linear fit, R^2 = {fit.rvalue**2:.4f}, n = {valid.sum()})"
    )
    return heating_coefficient, heating_coefficient_uncertainty

def _cooling_fit(data):
    time_seconds = data["Cooling Time"].to_numpy(dtype=float) * 60
    temperature_excess = (
        data["Cooling Temp bg"].to_numpy(dtype=float)
        - data["Cooling Temp"].to_numpy(dtype=float)
    )
    valid = np.isfinite(time_seconds) & np.isfinite(temperature_excess) & (temperature_excess > 0)
    if valid.sum() < 2:
        raise ValueError("At least two cooling measurements above ambient are required.")

    log_excess = np.log(temperature_excess[valid])
    return time_seconds[valid], log_excess, linregress(time_seconds[valid], log_excess)

def analyse_cooling(data):
    time_seconds, delta_t, fit = _cooling_fit(data)
    cooling_coefficient = -fit.slope
    cooling_coefficient_uncertainty = fit.stderr
    print(
        f"k = {cooling_coefficient:.3e} +/- {cooling_coefficient_uncertainty:.1e} s^-1 "
        f"(log-linear fit, R^2 = {fit.rvalue**2:.4f}, n = {len(time_seconds)})"
    )
    
    RMSE = np.sqrt(np.mean((delta_t - (fit.intercept + fit.slope * time_seconds))**2))
    print(f"RMSE = {RMSE:.3e} K")
    
    # Calculate b
    b = air_thermal_conductivity * calorimeter_surface_area / (cooling_coefficient * heat_capacity)
    b_uncertainty = b * np.sqrt(
        (calorimeter_surface_area_uncertainty / calorimeter_surface_area) ** 2
        + (cooling_coefficient_uncertainty / cooling_coefficient) ** 2
        + (heat_capacity_uncertainty / heat_capacity) ** 2
        + (RMSE / np.mean(delta_t))**2
    )
    print(f"Boundary-layer thickness = {b:.3e} {symbols['+/-']} {b_uncertainty:.1e} m")
    
    return cooling_coefficient, cooling_coefficient_uncertainty

def analyse_gamma(data):
    # Recalculate gamma from the periods and apparatus values, not the CSV's
    # precomputed gamma column, which does not match these measurements.
    periods = data["t"].to_numpy(dtype=float)
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if len(periods) < 2:
        raise ValueError("At least two valid oscillation periods are required.")

    mean_period = periods.mean()
    mean_period_uncertainty = periods.std(ddof=1) / np.sqrt(len(periods))
    gamma = (
        4 * np.pi**2 * ball_mass * enclosed_air_volume
        / (equilibrium_pressure * ball_cross_sectional_area**2 * mean_period**2)
    )
    relative_uncertainty = np.sqrt(
        (ball_mass_uncertainty / ball_mass) ** 2
        + (enclosed_air_volume_uncertainty / enclosed_air_volume) ** 2
        + (equilibrium_pressure_uncertainty / equilibrium_pressure) ** 2
        + (2 * ball_cross_sectional_area_uncertainty / ball_cross_sectional_area) ** 2
        + (2 * mean_period_uncertainty / mean_period) ** 2
    )
    gamma_uncertainty = gamma * relative_uncertainty
    print(
        f"gamma = {gamma:.3f} +/- {gamma_uncertainty:.3f} "
        f"(mean period = {mean_period:.4f} +/- {mean_period_uncertainty:.4f} s, n = {len(periods)})"
    )
    return gamma, gamma_uncertainty

def analyse_adiabatic_index(data):
    periods = data["t"].to_numpy(dtype=float)
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if len(periods) < 2:
        raise ValueError("At least two valid oscillation periods are required.")

    mean_period = periods.mean()
    mean_period_uncertainty = periods.std(ddof=1) / np.sqrt(len(periods))
    adiabatic_index = (
        4 * np.pi**2 * ball_mass * enclosed_air_volume
        / (equilibrium_pressure * ball_cross_sectional_area**2 * mean_period**2)
    )
    relative_uncertainty = np.sqrt(
        (ball_mass_uncertainty / ball_mass) ** 2
        + (enclosed_air_volume_uncertainty / enclosed_air_volume) ** 2
        + (equilibrium_pressure_uncertainty / equilibrium_pressure) ** 2
        + (2 * ball_cross_sectional_area_uncertainty / ball_cross_sectional_area) ** 2
        + (2 * mean_period_uncertainty / mean_period) ** 2
    )
    adiabatic_index_uncertainty = adiabatic_index * relative_uncertainty
    print(
        f"Adiabatic index = {adiabatic_index:.3f} +/- {adiabatic_index_uncertainty:.3f} "
        f"(mean period = {mean_period:.4f} +/- {mean_period_uncertainty:.4f} s, n = {len(periods)})"
    )
    return adiabatic_index, adiabatic_index_uncertainty

def plot_heating(data):
    fig, ax = plt.subplots(figsize=figure_size)
    sns.lineplot(x="Heating Time", y="Heating Temp", data=data, label="Heating Temp")
    sns.lineplot(x="Heating Time", y="Heating Temp bg", data=data, label="Heating Temp bg")
    Styling._style_axis(ax, "Heating Curve", "Time (min)", f"Temperature ({symbols['degree_celsius']})")
    Styling._finish_plot(fig, "Heating Curve", report_images_dir)

def plot_cooling(data):
    fig, ax = plt.subplots(figsize=figure_size)
    sns.lineplot(x="Cooling Time", y="Cooling Temp bg", data=data, label="Calorimeter Temp")
    sns.lineplot(x="Cooling Time", y="Cooling Temp", data=data, label="Ambient Temp")
    Styling._style_axis(ax, "Cooling Curve", "Time (min)", f"Temperature ({symbols['degree_celsius']})")
    Styling._finish_plot(fig, "Cooling Curve", report_images_dir)

def plot_delta_T_Heating(data):
    fig, ax = plt.subplots(figsize=figure_size)
    delta_T_heating = (data["Heating Temp"] - data["Heating Temp bg"])*-1
    sns.lineplot(x="Heating Time", y=delta_T_heating, data=data, label=f"{symbols['delta']}T Heating")
    Styling._style_axis(ax, "Temperature Difference Curve Heating", "Time (min)", f"{symbols['delta']}T ({symbols['degree_celsius']})")
    Styling._finish_plot(fig, "Temperature Difference Curve Heating", report_images_dir)

def plot_delta_T_Cooling(data):
    fig, ax = plt.subplots(figsize=figure_size)
    delta_T_cooling = data["Cooling Temp bg"] - data["Cooling Temp"]
    sns.lineplot(x="Cooling Time", y=delta_T_cooling, data=data, label=f"{symbols['delta']}T Cooling")
    Styling._style_axis(ax, "Temperature Difference Curve Cooling", "Time (min)", f"{symbols['delta']}T ({symbols['degree_celsius']})")
    ax.set_yscale("log")
    Styling._finish_plot(fig, "Temperature Difference Curve Cooling", report_images_dir)

def plot_cooling_log_fit(data):
    time_seconds, log_excess, fit = _cooling_fit(data)
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=figure_size, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.scatter(time_seconds, log_excess, label="Measured data", s=18)
    fit_time = np.linspace(time_seconds.min(), time_seconds.max(), 200)
    ax.plot(
        fit_time,
        fit.intercept + fit.slope * fit_time,
        color="tab:orange",
        label=(
            f"Linear fit: k = {-fit.slope:.3e} +/- {fit.stderr:.1e} s$^{{-1}}$"
            f"\n$R^2$ = {fit.rvalue**2:.4f}"
        ),
    )
    ax.legend()

    # Plot residuals
    residuals = log_excess - (fit.intercept + fit.slope * time_seconds)
    ax_res.scatter(time_seconds, residuals, s=18)
    ax_res.axhline(0, color="tab:orange", linestyle="--", label="Zero residual")
    ax_res.legend()
    Styling._style_axis(
        ax,
        "Logarithmic Cooling Fit",
        "Cooling time (s)",
        r"$\ln(T-T_0)$",
    )
    Styling._finish_plot(fig, "Cooling_Log_Linear_Fit", report_images_dir)

def plot_heating_rate_comparison(data):
    time_seconds = data["Heating Time"].to_numpy(dtype=float) * 60
    temperature_excess = (
        data["Heating Temp bg"].to_numpy(dtype=float)
        - data["Heating Temp"].to_numpy(dtype=float)
    )
    valid = (
        np.isfinite(time_seconds)
        & np.isfinite(temperature_excess)
        & (temperature_excess > 0)
    )
    if valid.sum() < 2:
        raise ValueError("At least two valid heating measurements above ambient are required.")

    time_seconds = time_seconds[valid]
    temperature_excess = temperature_excess[valid]
    fit = linregress(time_seconds, temperature_excess)
    _, _, cooling_fit = _cooling_fit(data)
    cooling_coefficient = -cooling_fit.slope
    if not np.isfinite(cooling_coefficient) or cooling_coefficient <= 0:
        raise ValueError("The fitted cooling coefficient must be positive.")

    heat_capacity = (
        calorimeter_mass * aluminium_specific_heat_capacity
        + water_mass * water_specific_heat_capacity
    )
    ideal_heating_rate = voltage * heating_current / heat_capacity
    fit_time = np.linspace(time_seconds.min(), time_seconds.max(), 200)
    elapsed_time = fit_time - time_seconds[0]
    cooling_aware_temperature_excess = (
        temperature_excess[0] * np.exp(-cooling_coefficient * elapsed_time)
        + ideal_heating_rate / cooling_coefficient
        * (1 - np.exp(-cooling_coefficient * elapsed_time))
    )
    no_loss_temperature_excess = (
        temperature_excess[0] + ideal_heating_rate * elapsed_time
    )
    measured_cooling_aware_temperature_excess = (
        temperature_excess[0]
        * np.exp(-cooling_coefficient * (time_seconds - time_seconds[0]))
        + ideal_heating_rate / cooling_coefficient
        * (1 - np.exp(-cooling_coefficient * (time_seconds - time_seconds[0])))
    )

    fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=figure_size, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    ax.scatter(time_seconds, temperature_excess, label="Measured temperature excess", s=18)
    ax.plot(
        fit_time,
        fit.intercept + fit.slope * fit_time,
        color="tab:orange",
        label=f"Measured linear fit: {fit.slope:.3e} K s$^{{-1}}$",
    )
    ax.plot(
        fit_time,
        cooling_aware_temperature_excess,
        color="tab:green",
        linewidth=2,
        label=(
            "Constant-power model with cooling "
            f"($k={cooling_coefficient:.3e}$ s$^{{-1}}$)"
        ),
    )
    ax.plot(
        fit_time,
        no_loss_temperature_excess,
        color="tab:purple",
        linestyle=":",
        label=f"No-loss reference, $VI/C$: {ideal_heating_rate:.3e} K s$^{{-1}}$",
    )
    ax.legend()
    
    residuals = temperature_excess - (fit.intercept + fit.slope * time_seconds)
    ax_res.scatter(time_seconds, residuals, s=18)
    ax_res.axhline(0, color="tab:orange", linestyle="--", label="Zero residual")
    ax_res.legend()
    
    Styling._style_axis(
        ax,
        "Heating with Simultaneous Cooling",
        "Heating time (s)",
        r"Temperature excess, $T-T_0$ ($^\circ$C)",
    )
    Styling._finish_plot(fig, "Heating_With_Cooling", report_images_dir)
    
def plot_adiabatic(data):
    fig, ax = plt.subplots(figsize=figure_size)
    sns.scatterplot(x=data["n"], y=data["T"], data=data)
    regress = (linregress(data["n"], data["T"]))
    t = regress.slope/8
    c = regress.intercept
    x = range(12)
    ax.plot(x, 8*t*x + c, label=f"Fit: t = {t:.2f}s, c = {c:.2f}")
    ax.legend()
    Styling._style_axis(ax, "Period vs n", "n", f"Period (s/8)")
    Styling._finish_plot(fig, "Period_vs_n_adiabatic", report_images_dir)

if __name__ == "__main__":
    choice = input("Which analysis do you want? (c)ooling, (h)eating, (a)diabatic, (g)amma or (G)raphs: ").strip()
    if choice == "c":
        analyse_cooling(data)
    elif choice == "h":
        analyse_heating(data)
    elif choice == "g":
        analyse_gamma(data_drops)
    elif choice == "a":
        analyse_adiabatic_index(data_drops)
    elif choice == "G":
        #plot_heating(data)
        #plot_cooling(data)
        #plot_delta_T_Heating(data)
        #plot_delta_T_Cooling(data)
        plot_cooling_log_fit(data)
        plot_heating_rate_comparison(data)
        plot_adiabatic(data_drops)
