from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
import Styling

wavelength_cmap = sns.color_palette("crest", as_cmap=True)
wavelength_cmap = wavelength_cmap.reversed()
plot_colors = sns.color_palette("flare", 4)
figure_size = (8, 5)

uncertainties = {
    "x": 2.0e-3,
    "L": 0.5e-3
}


def exponential(x, a, b, c):
    return 1 / (b* (x + a)) + c

def sigmoid(voltage, current_low, current_sat, voltage_mid, voltage_scale):
    return current_low + (current_sat - current_low) / (
        1 + np.exp(-(voltage - voltage_mid) / voltage_scale)
    )

data_slits = pd.read_csv("Code/Data/Lights_slit.csv")
data_trends = pd.read_csv("Code/Data/Lights_trends.csv")
data_stops = pd.read_csv("Code/Data/Lights.csv")
data_volts_v_amps = pd.read_csv("Code/Data/Lights_versus.csv")

colours_primary = ["blue", "teal", "green", "orange", "red"]

report_images_dir = "Reports/Report 4 - Light/images"

slits_d = 0.001/600

def plot_trends(data):
    fig, ax = plt.subplots(figsize=figure_size)
    points = ax.scatter(
        data["intensity"],
        data["Average_V_max"],
        c=data["wavelength"],
        cmap=wavelength_cmap
    )
    Styling._style_axis(ax, "Average V_max vs Intensity", "Intensity", "Average V_max")

    # Fit each group by wavelength using exponential fit
    for wavelength in data["wavelength"].unique():
        subset = data[data["wavelength"] == wavelength]
        # Exponenetial fit
        popt, _ = curve_fit(exponential, subset["intensity"], subset["Average_V_max"])
        ax.plot(
            subset["intensity"],
            exponential(subset["intensity"], *popt),
            label=f"Wavelength {wavelength}, a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}",
            linestyle="--"
        )
    ax.legend()
    ax.set_yscale("log")
    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("Wavelength")
    Styling._finish_plot(fig, "trends_plot.png", report_images_dir)

def plot_slits(data):
    if len(data) != 20:
        raise ValueError(f"Expected 20 slit measurements (five lights x four readings), got {len(data)}")

    fig, ax = plt.subplots(figsize=figure_size)
    line_orders = np.linspace(0, data["n"].max(), 100)

    # The CSV has no light identifier, so each consecutive block of four rows is one light.
    light_fits = []
    for light_number, start in enumerate(range(0, len(data), 4), start=1):
        measurements = data.iloc[start:start + 4]
        orders = measurements["n"].to_numpy()
        sin_theta = measurements["sin theta"].to_numpy()
        slope = np.dot(orders, sin_theta) / np.dot(orders, orders)
        wavelength_nm = slope * slits_d * 1e9
        light_fits.append((light_number, orders, sin_theta, slope, wavelength_nm))

    wavelengths = [fit[4] for fit in light_fits]
    wavelength_min, wavelength_max = min(wavelengths), max(wavelengths)
    if wavelength_min == wavelength_max:
        wavelength_max = wavelength_min + 1
    wavelength_range = wavelength_max - wavelength_min

    for light_number, orders, sin_theta, slope, wavelength_nm in light_fits:
        color = wavelength_cmap((wavelength_nm - wavelength_min) / wavelength_range)
        ax.plot(
            line_orders,
            slope * line_orders,
            label=f"Light {light_number} (λ ≈ {wavelength_nm:.0f} nm)",
            color=color
        )

    scatter_orders = np.concatenate([fit[1] for fit in light_fits])
    scatter_sin_theta = np.concatenate([fit[2] for fit in light_fits])
    scatter_wavelengths = np.concatenate([
        np.full(fit[1].shape, fit[4]) for fit in light_fits
    ])
    points = ax.scatter(
        scatter_orders,
        scatter_sin_theta,
        c=scatter_wavelengths,
        cmap=wavelength_cmap,
        vmin=wavelength_min,
        vmax=wavelength_max,
        s=35
    )

    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Wavelength (nm)")

    Styling._style_axis(
        ax,
        "Diffraction measurements for five lights",
        "Diffraction order, n",
        "sin(θ)"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    Styling._finish_plot(fig, "slits_plot.png", report_images_dir)
    
    # Save the new frequencies and errors
    new_data = []
    for light_number, start in enumerate(range(0, len(data), 4), start=1):
        measurements = data.iloc[start:start + 4]
        orders = measurements["n"].to_numpy()
        sin_theta = measurements["sin theta"].to_numpy()
        slope = np.dot(orders, sin_theta) / np.dot(orders, orders)
        wavelength_nm = slope * slits_d * 1e9
        residuals = sin_theta - slope * orders
        slope_se = np.sqrt(
            np.sum(residuals**2)
            / ((len(orders) - 1) * np.dot(orders, orders))
        )
        u_wavelength_nm = slope_se * slits_d * 1e9
        new_data.append({
            "Light": light_number,
            "Wavelength (nm)": wavelength_nm,
            "Uncertainty (nm)": u_wavelength_nm
        })
    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_csv(f"Code/Data/slits_new_frequencies.csv", index=False)
        
def plot_stopping_voltage(data):
    frequency = data["frequency"].to_numpy()
    stopping_energy = data["V_avg"].to_numpy() * 1.60218e-19  # Convert eV to J

    slope, intercept = np.polyfit(frequency, stopping_energy, 1)
    if slope == 0:
        raise ValueError("Cannot determine a threshold frequency from a zero-slope fit")

    threshold_frequency = -intercept / slope
    work_function = -intercept
    fit_frequencies = np.linspace(
        0, max(frequency.max(), threshold_frequency), 200
    )

    # Set the Seaborn colormap
    cmap = wavelength_cmap
    point_colors = cmap(
        np.linspace(0, 1, len(frequency))
    )

    fig, ax = plt.subplots(figsize=figure_size)
    
    errors = [8.0E-23, 2.3E-21, 7.2E-22, 3.4E-21, 3.4E-21]

    points = ax.scatter(
        frequency,
        stopping_energy,
        c=point_colors,
        label="Measurements",
        zorder=3
    )
    
    ax.errorbar(
        frequency,
        stopping_energy,
        yerr=errors,
        fmt='o',
        color='black',
        ecolor='black',
        elinewidth=2,
        capsize=4,
        zorder=2
    )

    # Positive fitted line: solid
    ax.plot(
        np.linspace(threshold_frequency, fit_frequencies[-1], 200),
        slope * np.linspace(threshold_frequency, fit_frequencies[-1], 200) + intercept,
        color=plot_colors[1],
        linestyle="-",
        label=(
            f"Fit: h = {slope:.2e} J·s, "
            f"φ = {work_function:.2e} J"
        )
    )

    # Horizontal threshold construction: solid
    ax.plot(
        [0, threshold_frequency],
        [0, 0],
        color=plot_colors[1],
        linestyle="-"
    )

    # Negative extrapolation to the y-axis: dashed
    ax.plot(
        [0, threshold_frequency],
        [intercept, 0],
        color=plot_colors[3],
        linestyle="--"
    )

    # Mark and annotate the y-intercept
    ax.scatter(
        [0], [intercept],
        color=plot_colors[3],
        zorder=4,
        label=f"Y-intercept ≈ {intercept:.2e} J"
    )
    

    ax.annotate(
        f"(0, {intercept:.2e} J)",
        xy=(0, intercept),
        xytext=(8, -12),
        textcoords="offset points",
        color=plot_colors[3]
    )

    Styling._style_axis(
        ax,
        "Stopping Energy vs Frequency",
        "Frequency (Hz)",
        "Stopping energy (J)"
    )
    ax.set_xlim(left=0)
    ax.legend()
    Styling._finish_plot(fig, "stopping_voltage_plot.png", report_images_dir)

    
def plot_volts_v_amps(data):
    # Plot the measured photocurrent versus electrode voltage and fit a sigmoid curve and the residuals below on same plot and show entire fit
    volts = -data["Voltage"].to_numpy()
    amps = -data["Current"].to_numpy()

    parameters, _ = curve_fit(
        sigmoid,
        volts,
        amps,
        p0=[amps.min() - 1.75, amps.max(), 0.0, 0.25],
        maxfev=10000
    )
    fit_volts = np.linspace(volts.min(), -volts.min(), 300)
    residuals = amps - sigmoid(volts, *parameters)
    r_squared = 1 - np.sum(residuals**2) / np.sum((amps - amps.mean())**2)
    
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.scatter(volts, amps, color=plot_colors[0], label="Measurements")
    ax.plot(
        fit_volts,
        sigmoid(fit_volts, *parameters),
        color=plot_colors[1],
        linestyle="--",
        label=f"Sigmoid fit ($R^2={r_squared:.5f}$)"
    )
    ax.set_xlim(left=volts.min() * 1.05, right=-volts.min() * 1.05)
    ax.set_ylim(bottom=-1, top=amps.max() * 2.5)
    ax.set_title("Current vs Voltage")
    ax.set_ylabel("Current (nA)")
    ax.legend()

    ax_res.scatter(volts, residuals, color=plot_colors[0], label="Residuals")
    ax_res.axhline(0, color=plot_colors[1], linestyle="--")
    ax_res.set_xlim(left=volts.min() * 1.05, right=-volts.min() * 1.05)
    ax_res.set_xlabel("Voltage (V)")
    ax_res.set_ylabel("Residual Current (nA)")
    ax_res.legend()

    print(f"Fitted parameters: {parameters}")
    Styling._finish_plot(fig, "volts_v_amps_plot.png", report_images_dir)    

if __name__ == "__main__":
    choice = input("Enter the plot you want to generate 1-5(slits, stopping_voltage, trends, volts_v_amps, all): ").strip().lower()
    if choice == "1":
        choice = "slits"
    elif choice == "2":
        choice = "stopping_voltage"
    elif choice == "3":
        choice = "trends"
    elif choice == "4":
        choice = "volts_v_amps"
    elif choice == "5":
        choice = "all"
    if choice == "slits":
        plot_slits(data_slits)
    elif choice == "stopping_voltage":
        plot_stopping_voltage(data_stops)
    elif choice == "trends":
        plot_trends(data_trends)
    elif choice == "volts_v_amps":
        plot_volts_v_amps(data_volts_v_amps)
    elif choice == "all":
        plot_slits(data_slits)
        plot_stopping_voltage(data_stops)
        plot_trends(data_trends)
        plot_volts_v_amps(data_volts_v_amps)
    else:
        print("Invalid choice.")