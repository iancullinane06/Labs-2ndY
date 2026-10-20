import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

sns.set_style('dark')


# ============================================================
# Import data
# ============================================================

data = pd.read_csv("spring_experiment.csv")


# ============================================================
# Get the time and magnetic field data
# ============================================================

time_run2 = data["Latest: Time (s)"].to_numpy(dtype=float)
mag_run2 = data["Latest: Magnetic Field (mT)"].to_numpy(dtype=float)

time_run1 = data["Run 1: Time (s)"].to_numpy(dtype=float)
mag_run1 = data["Run 1: Magnetic Field (mT)"].to_numpy(dtype=float)


# ============================================================
# Calculate sampling intervals
# ============================================================

dt_run2 = np.mean(np.diff(time_run2))
dt_run1 = np.mean(np.diff(time_run1))

print("Run 2 dt =", dt_run2, "s")
print("Run 1 dt =", dt_run1, "s")

print("Run 2 sampling frequency =", 1 / dt_run2, "Hz")
print("Run 1 sampling frequency =", 1 / dt_run1, "Hz")


# ============================================================
# Remove DC component
# ============================================================

mag_run2 = mag_run2 - np.mean(mag_run2)
mag_run1 = mag_run1 - np.mean(mag_run1)


# ============================================================
# Calculate FFTs
# ============================================================

fft_run2 = np.fft.rfft(mag_run2)
fft_run1 = np.fft.rfft(mag_run1)


# ============================================================
# Calculate frequencies
# ============================================================

freq_run2 = np.fft.rfftfreq(len(mag_run2), d=dt_run2)
freq_run1 = np.fft.rfftfreq(len(mag_run1), d=dt_run1)


# ============================================================
# Calculate FFT magnitude
# ============================================================

amp_run2 = np.abs(fft_run2)
amp_run1 = np.abs(fft_run1)


# ============================================================
# Limit analysis to 1–10 Hz
# ============================================================

mask_run2 = (freq_run2 >= 1) & (freq_run2 <= 10)
mask_run1 = (freq_run1 >= 1) & (freq_run1 <= 10)

f_run2 = freq_run2[mask_run2]
a_run2 = amp_run2[mask_run2]

f_run1 = freq_run1[mask_run1]
a_run1 = amp_run1[mask_run1]


# ============================================================
# Find dominant frequency
# ============================================================

peak_run2_index = np.argmax(a_run2)
peak_run1_index = np.argmax(a_run1)

peak_freq_run2 = f_run2[peak_run2_index]
peak_freq_run1 = f_run1[peak_run1_index]

print()
print(f"Run 2 peak frequency = {peak_freq_run2:.3f} Hz")
print(f"Run 1 peak frequency = {peak_freq_run1:.3f} Hz")


# ============================================================
# Gaussian function
#
# A     = peak height
# mu    = centre frequency
# sigma = width
# ============================================================

def gaussian(f, A, mu, sigma):
    return A * np.exp(
        -0.5 * ((f - mu) / sigma) ** 2
    )


# ============================================================
# Select region around each peak for Gaussian fitting
# ============================================================

# This determines how much of the peak is used in the fit.
# Increase this if you want the Gaussian to include more
# of the broad shoulders.
#
# 0.15 Hz means ±0.15 Hz around the peak.

fit_width = 0.15


mask_fit_run2 = (
    (f_run2 >= peak_freq_run2 - fit_width) &
    (f_run2 <= peak_freq_run2 + fit_width)
)

mask_fit_run1 = (
    (f_run1 >= peak_freq_run1 - fit_width) &
    (f_run1 <= peak_freq_run1 + fit_width)
)


# ============================================================
# Initial guesses for Gaussian fitting
# ============================================================

p0_run2 = [
    np.max(a_run2),      # A
    peak_freq_run2,      # mu
    0.05,                # sigma
]

p0_run1 = [
    np.max(a_run1),      # A
    peak_freq_run1,      # mu
    0.05,                # sigma
]


# ============================================================
# Fit Gaussian to Run 2
# ============================================================

params_run2, covariance_run2 = curve_fit(
    gaussian,
    f_run2[mask_fit_run2],
    a_run2[mask_fit_run2],
    p0=p0_run2,
    bounds=(
        [0, 1, 0.0001],
        [np.inf, 4.5, 1]
    )
)

A2, mean_run2, std_run2 = params_run2


# ============================================================
# Fit Gaussian to Run 1
# ============================================================

params_run1, covariance_run1 = curve_fit(
    gaussian,
    f_run1[mask_fit_run1],
    a_run1[mask_fit_run1],
    p0=p0_run1,
    bounds=(
        [0, 1, 0.0001],
        [np.inf, 4.5, 1]
    )
)

A1, mean_run1, std_run1 = params_run1


# ============================================================
# Calculate uncertainties in fitted parameters
# ============================================================

parameter_errors_run2 = np.sqrt(np.diag(covariance_run2))
parameter_errors_run1 = np.sqrt(np.diag(covariance_run1))

A2_error = parameter_errors_run2[0]
mean_run2_error = parameter_errors_run2[1]
std_run2_error = parameter_errors_run2[2]
#C2_error = parameter_errors_run2[3]

A1_error = parameter_errors_run1[0]
mean_run1_error = parameter_errors_run1[1]
std_run1_error = parameter_errors_run1[2]
#C1_error = parameter_errors_run1[3]


# ============================================================
# Calculate FWHM from fitted sigma
#
# For a Gaussian:
#
# FWHM = 2.355 * sigma
# ============================================================

fwhm_run2 = 2.355 * std_run2
fwhm_run1 = 2.355 * std_run1


# ============================================================
# Print results
# ============================================================

print()
print("========================================")
print("RUN 2 RESULTS")
print("========================================")

print(f"Peak frequency = {mean_run2:.4f} Hz")
print(f"Frequency uncertainty = ±{mean_run2_error:.4f} Hz")

print(f"Gaussian sigma = {std_run2:.4f} Hz")
print(f"Sigma uncertainty = ±{std_run2_error:.4f} Hz")

print(f"FWHM = {fwhm_run2:.4f} Hz")

print(f"±2σ = {2 * std_run2:.4f} Hz")

print()
print("========================================")
print("RUN 1 RESULTS")
print("========================================")

print(f"Peak frequency = {mean_run1:.4f} Hz")
print(f"Frequency uncertainty = ±{mean_run1_error:.4f} Hz")

print(f"Gaussian sigma = {std_run1:.4f} Hz")
print(f"Sigma uncertainty = ±{std_run1_error:.4f} Hz")

print(f"FWHM = {fwhm_run1:.4f} Hz")

print(f"±2σ = {2 * std_run1:.4f} Hz")


# ============================================================
# Generate smooth Gaussian curves
# ============================================================

x = np.linspace(1, 4.5, 1000)

normal_run2 = gaussian(
    x,
    A2,
    mean_run2,
    std_run2
)

normal_run1 = gaussian(
    x,
    A1,
    mean_run1,
    std_run1
)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 6))


# ------------------------------------------------------------
# FFT data
# ------------------------------------------------------------

plt.plot(
    f_run2,
    a_run2,
    color="orange",
    label="Run 2 FFT"
)

plt.plot(
    f_run1,
    a_run1,
    color="blue",
    label="Run 1 FFT"
)


# ------------------------------------------------------------
# Gaussian fits
# ------------------------------------------------------------

plt.plot(
    x,
    normal_run2,
    "--",
    color="orange",
    linewidth=2,
    label="Run 2 Gaussian Fit"
)

plt.plot(
    x,
    normal_run1,
    "--",
    color="blue",
    linewidth=2,
    label="Run 1 Gaussian Fit"
)


# ------------------------------------------------------------
# Mean / centre frequencies
# ------------------------------------------------------------

plt.axvline(
    mean_run2,
    color="orange",
    linestyle=":",
    linewidth=2,
    label=f"Run 2 Centre = {mean_run2:.2f} Hz"
)

plt.axvline(
    mean_run1,
    color="blue",
    linestyle=":",
    linewidth=2,
    label=f"Run 1 Centre = {mean_run1:.2f} Hz"
)


# ============================================================
# Formatting
# ============================================================

plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT Magnitude")

plt.title("FFT Comparison: Run 1 vs Run 2")

plt.xlim(1, 10)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()