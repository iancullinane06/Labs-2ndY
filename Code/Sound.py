from math import pi, sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from labplotlib import (
    Measurement,
    apply_theme,
    finish_plot,
    polynomial_fit,
    scatter_with_fit,
    scatter_with_residuals,
)

apply_theme()

REPORT_DIR = "Reports/Report 8 - Sound/images"
END_CORRECTION = 0.01326
SOUND_SPEED_ESTIMATES = (
    ("Fixed length", 343.43, 3.38),
    ("Time of flight", 347.69, 3.80),
    (r"$\Delta L$", 339.20, 4.69),
    ("Constant frequency", 367.84, 33.79),
)

data_constant_length = pd.read_csv("Code/Data/Sound_Length.csv")
data_constant_frequency = pd.read_csv("Code/Data/Sound_Frequency.csv")
data_young = pd.read_csv("Code/Data/Sound_Young.csv")
data_delta_l = pd.read_csv("Code/Data/Sound_Delta_L.csv")
data_speed_of_travel = pd.read_csv("Code/Data/Sound_Speed_of_Travel.csv")

def analyse_constant_length(data):
    frequency = Measurement.from_dataframe(data, "Frequency", granularity=5)
    length = Measurement.from_pm(data["Length"].mean(), 0.01)
    harmonics = Measurement.from_dataframe(data, "harmonic")
    c = frequency * length * 4 / (2*harmonics + 1)
    c.set_unit("m/s")
    print(f"c_calculated = {c.mean()}")
    return frequency, c

def analyse_constant_frequency(data):
    length_correction = Measurement.from_pm(END_CORRECTION, 0.000005)
    frequency = Measurement.from_dataframe(data, "Frequency", granularity=5)
    length = Measurement.from_dataframe(data, "Length", uncertainty="pm L", granularity=0.01)
    harmonics = Measurement.from_dataframe(data, "harmonic")
    c = frequency * (length + length_correction) * 4 / (2*harmonics - 1)
    c.set_unit("m/s")
    print(f"c_calculated = {c.mean()}")
    return frequency, c

def analyse_speed_of_travel(data):
    length = Measurement.from_dataframe(data, "Length", granularity=0.001)
    time = Measurement.from_dataframe(data, "Time", granularity=0.00001)
    c = length / time
    c.set_unit("m/s")
    print(f"c_calculated = {c.mean()}")
    return length, time, c

def analyse_delta_l(data):
    frequency = Measurement.from_dataframe(data, "Frequency", granularity=100)
    delta_l = Measurement.from_dataframe(data, "Delta L", granularity=0.001)
    c = delta_l * frequency * 2
    c.set_unit("m/s")
    print(f"c_calculated = {c.mean()}")
    return frequency, c

def analyse_youngs(data):
    frequency = Measurement.from_dataframe(data, "frequency", granularity=10)
    fit = polynomial_fit(data["n"], frequency, degree=1)
    lambda_measurement = Measurement.from_pm(1.536, 0.002)
    c = fit.coefficients[0]*2*lambda_measurement
    print(f"c_brass = {c}")
    brass = Measurement.from_pm(8.6e3, 0.2e3)
    Y = c**2 * brass
    print(f"Y = {Y.to_string(unit='Pa', prefix='giga')}")
    return Y


def analyse_shear(Y):
    # Calculate shear modulus from Young's modulus
    nu = Measurement.from_pm(0.34, 0.01)  # Poisson's ratio for brass
    G = Y / (2 * (1 + nu))
    print(f"G = {G.to_string(unit='Pa', prefix='giga')}")
    return G


def _combine_sound_speed_estimates():
    weights = [1 / uncertainty**2 for _, _, uncertainty in SOUND_SPEED_ESTIMATES]
    total_weight = sum(weights)
    mean = sum(
        weight * estimate
        for weight, (_, estimate, _) in zip(weights, SOUND_SPEED_ESTIMATES)
    ) / total_weight
    uncertainty = sqrt(1 / total_weight)
    chi_squared = sum(
        ((estimate - mean) / estimate_uncertainty) ** 2
        for _, estimate, estimate_uncertainty in SOUND_SPEED_ESTIMATES
    )
    return mean, uncertainty, chi_squared


def Plot_Sound_Speed_Consistency(show=False):
    combined_mean, combined_uncertainty, chi_squared = (
        _combine_sound_speed_estimates()
    )
    figure, (pdf_axis, result_axis) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    speeds = np.linspace(320, 385, 800)
    colors = []
    for label, estimate, uncertainty in SOUND_SPEED_ESTIMATES:
        density = np.exp(-0.5 * ((speeds - estimate) / uncertainty) ** 2)
        density /= uncertainty * sqrt(2 * pi)
        line, = pdf_axis.plot(
            speeds, density, label=f"{label}: {estimate:.2f} ± {uncertainty:.2f}"
        )
        colors.append(line.get_color())

    combined_density = np.exp(
        -0.5 * ((speeds - combined_mean) / combined_uncertainty) ** 2
    ) / (combined_uncertainty * sqrt(2 * pi))
    pdf_axis.plot(
        speeds,
        combined_density,
        color="black",
        linewidth=2.5,
        label=f"Combined: {combined_mean:.2f} ± {combined_uncertainty:.2f}",
    )
    pdf_axis.fill_between(
        speeds,
        combined_density,
        where=np.abs(speeds - combined_mean) <= combined_uncertainty,
        color="black",
        alpha=0.12,
    )
    pdf_axis.set(
        title="Normal distributions (central speed range)",
        xlabel="Speed of sound (m/s)",
        ylabel="Probability density (s/m)",
    )
    pdf_axis.legend(fontsize="x-small")

    results = list(SOUND_SPEED_ESTIMATES) + [
        ("Combined", combined_mean, combined_uncertainty)
    ]
    positions = np.arange(len(results))
    for position, (label, estimate, uncertainty) in enumerate(results):
        color = "black" if label == "Combined" else colors[position]
        result_axis.errorbar(
            estimate,
            position,
            xerr=uncertainty,
            fmt="D" if label == "Combined" else "o",
            color=color,
            capsize=3,
        )
    result_axis.axvline(
        combined_mean, color="black", linestyle="--", linewidth=1, alpha=0.6
    )
    result_axis.set(
        title=rf"Combined fit: $\chi^2={chi_squared:.2f}$ (3 dof)",
        xlabel="Speed of sound (m/s)",
        yticks=positions,
        yticklabels=[label for label, _, _ in results],
        xlim=(
            min(estimate - 2 * uncertainty for _, estimate, uncertainty in results),
            max(estimate + 2 * uncertainty for _, estimate, uncertainty in results),
        ),
    )
    result_axis.invert_yaxis()

    finish_plot(
        figure, f"{REPORT_DIR}/sound_speed_consistency.png", show=show
    )
    plt.close(figure)


def Plot_Constant_Length(data, show=False):
    frequency = Measurement.from_dataframe(data, "Frequency", granularity=5)
    mode = 2 * data["harmonic"] + 1
    fig = scatter_with_residuals(mode, frequency,
                                xlabel="Odd mode number, n",
                                ylabel="Frequency (Hz)",
                                title="Closed-tube resonances at fixed length")
    finish_plot(fig, f"{REPORT_DIR}/sound_constant_length.png", show=show)


def Plot_Constant_Frequency(data, show=False):
    fig = None
    ax = None
    for frequency, group in data.groupby("Frequency"):
        group_modes = 2 * group["harmonic"].to_numpy() - 1
        effective_lengths = (
            Measurement.from_dataframe(group, "Length", uncertainty="pm L", granularity=0.01)
            + Measurement.from_pm(END_CORRECTION, 0.000005)
        )
        fig = scatter_with_fit(
            group_modes,
            effective_lengths,
            xlabel="Odd mode number, n",
            ylabel="Effective resonance length (m)",
            title="Resonance length at fixed frequency",
            label=f"{frequency:g} Hz",
            label_slope=True,
            ax=ax,
        )
        ax = fig.axes[0]
    ax.legend(title="Frequency")
    finish_plot(fig, f"{REPORT_DIR}/sound_constant_frequency.png", show=show)


def Plot_Delta_L(data, show=False):
    inverse_frequency = 1 / data["Frequency"]
    delta_l = data["Delta L"]
    fig = scatter_with_fit(
        inverse_frequency,
        delta_l,
        xlabel="Inverse frequency (s)",
        ylabel=r"$\Delta L$ (m)",
        title="Resonance spacing against inverse frequency",
        label="Measured",
    )
    finish_plot(fig, f"{REPORT_DIR}/sound_delta_l.png", show=show)


def Plot_Youngs(data, show=False):
    frequency = Measurement.from_dataframe(data, "frequency", granularity=10)
    fig = scatter_with_residuals(data["n"], frequency,
                                xlabel="n", ylabel="frequency (Hz)",
                                title="Longitudinal resonances of the brass rod")
    finish_plot(fig, f"{REPORT_DIR}/Sound_Young_experiment.png", show=show)

def Plot_Speed_of_Travel(data, show=False):
    length = Measurement.from_dataframe(data, "Length", granularity=0.001)
    time = Measurement.from_dataframe(data, "Time", granularity=0.00001)
    c = length / time
    c.set_unit("m/s")
    fig = scatter_with_fit(
        time,
        length,
        xlabel="Time (s)",
        ylabel="Length (m)",
        title="Speed of sound from time-of-flight measurements",
        label="Calculated",
    )
    finish_plot(fig, f"{REPORT_DIR}/sound_speed_of_travel.png", show=show)

if __name__ == "__main__":
    choice = input("Constant length, Constant frequency, Delta L, Speed of travel, moduli, comparison or All? (l/f/d/s/m/c/a): ").strip().lower()
    if choice == "l":
        analyse_constant_length(data_constant_length)
        Plot_Constant_Length(data_constant_length)
    elif choice == "f":
        analyse_constant_frequency(data_constant_frequency)
        Plot_Constant_Frequency(data_constant_frequency)
    elif choice == "d":
        analyse_delta_l(data_delta_l)
        Plot_Delta_L(data_delta_l)
    elif choice == "s":
        analyse_speed_of_travel(data_speed_of_travel)
        Plot_Speed_of_Travel(data_speed_of_travel)
    elif choice == "m":
        Y = analyse_youngs(data_young)
        G = analyse_shear(Y)
    elif choice == "c":
        Plot_Sound_Speed_Consistency()
    elif choice == "a":
        print("Running all analyses...")
        print("Starting constant length analysis...")
        analyse_constant_length(data_constant_length)
        Plot_Constant_Length(data_constant_length)
        print("Starting constant frequency analysis...")
        analyse_constant_frequency(data_constant_frequency)
        Plot_Constant_Frequency(data_constant_frequency)
        print("Starting delta L analysis...")
        analyse_delta_l(data_delta_l)
        Plot_Delta_L(data_delta_l)
        print("Starting modulii analysis...")
        Plot_Youngs(data_young)
        G = analyse_shear(analyse_youngs(data_young))
        print("Starting speed of travel analysis...")
        analyse_speed_of_travel(data_speed_of_travel)
        Plot_Speed_of_Travel(data_speed_of_travel)
        Plot_Sound_Speed_Consistency()
    else:
        print("Invalid choice.")