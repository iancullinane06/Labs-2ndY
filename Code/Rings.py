
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import seaborn as sns

sns.set_style(style="darkgrid")

# Newton's rings graph
# X axis: ring index n
# Y axis: r^2/R with associated uncertainty


def analyse_and_plot(file, name_of_plot):
    # Open file
    file_path = file
    data = pd.read_csv(file_path)

    # Constants and uncertainties
    R = 0.2625          # Radius of curvature (m)
    delta_R = 0.008     # Uncertainty in R (m)

    # Uncertainty in r (4 pixels converted to metres)
    delta_r = 1.24815e-5  # m

    # Extract data
    x = data['n'].to_numpy()
    y = data['r^2/R'].values

    # Recover r^2 and r
    r2 = y * R
    r = np.sqrt(r2)

    # Propagate uncertainty from r to r^2:
    # delta(r^2) = 2*r*delta_r
    delta_r2 = 2 * r * delta_r

    # Propagate uncertainty from r^2 to y = r^2/R
    # Keep R uncertainty out of the regression because R is common
    # to all measurements.
    delta_y = delta_r2 / R
    # Linear regression: y = slope*x + intercept
    def linear_model(x, slope, intercept):
        return slope * x + intercept

    # Weighted least-squares regression
    # Weights are based on the uncertainty in y
    popt, pcov = curve_fit(
        linear_model,
        x,
        y,
        sigma=delta_y,
        absolute_sigma=True
    )

    slope, intercept = popt

    # Uncertainties in slope and intercept
    slope_error, intercept_error = np.sqrt(np.diag(pcov))

    # Print regression results
    print(f"Slope: {slope:.6e} +/- {slope_error:.6e} m")
    print(f"Intercept: {intercept:.6e} +/- {intercept_error:.6e} m")

    # Wavelength from the slope
    wavelength = slope
    wavelength_error_fit = slope_error

    # Propagate the uncertainty in R separately
    # Since R is common to all measurements, it is correlated
    # and should not be counted as independent scatter.

    wavelength_error_R = abs(wavelength) * delta_R / R

    # Combine independent fit uncertainty and R uncertainty
    wavelength_error_total = np.sqrt(
        wavelength_error_fit**2 + wavelength_error_R**2
    )

    print(f"Wavelength: {wavelength * 1e9:.3f} nm")
    print(f"Fit uncertainty: {wavelength_error_fit * 1e9:.3f} nm")
    print(f"R uncertainty contribution: {wavelength_error_R * 1e9:.3f} nm")
    print(f"Total uncertainty: {wavelength_error_total * 1e9:.3f} nm")

    # Plot data with error bars
    plt.figure(figsize=(8, 6))

    plt.errorbar(
        x, y,
        yerr=delta_y,
        fmt='o',
        capsize=4,
        label='Data with uncertainty'
    )

    # Regression line
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = linear_model(x_fit, slope, intercept)

    plt.plot(
        x_fit, y_fit,
        color='teal',
        label='Weighted linear regression'
    )

    # Approximate 95% confidence band for the fitted mean
    X = np.vstack([x_fit, np.ones_like(x_fit)]).T
    prediction_error = np.sqrt(
        np.einsum('ij,jk,ik->i', X, pcov, X)
    )

    plt.fill_between(
        x_fit,
        y_fit - 1.96 * prediction_error,
        y_fit + 1.96 * prediction_error,
        color='teal',
        alpha=0.2,
        label='95% confidence band (fit)'
    )

    # Regression values on chart
    plt.text(
        0.05, 0.95,
        f"Slope: {slope:.6e} +/- {slope_error:.6e} m\nIntercept: {intercept:.6e} +/- {intercept_error:.6e} m",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    

    plt.xlabel('Ring index n')
    plt.ylabel(r'$r^2/R$ (m)')
    plt.title("Newton's Rings: Weighted Linear Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(name_of_plot)
    plt.show()
    
    return wavelength, wavelength_error_total, slope, intercept

def plot_residuals(file_name, slope, intercept, graph_name):
    data = pd.read_csv(file_name)
    x = data['n'].to_numpy()
    y = data['r^2/R'].to_numpy()

    residuals = y - (slope * x + intercept)
    R = 0.2625
    delta_r = 1.24815e-5
    delta_y = 2 * np.sqrt(y * R) * delta_r / R
    standardized_residuals = residuals / delta_y

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x, standardized_residuals,
        fmt='o',
        capsize=4,
        label='Standardized residuals'
    )
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(2, color='red', linestyle=':', label=r'$\pm2$ uncertainty')
    plt.axhline(-2, color='red', linestyle=':')
    plt.xlabel('Ring index n')
    plt.ylabel(r'Standardised residual $(y-y_{\mathrm{fit}})/\sigma_y$')
    plt.title("Standardised Residuals of Newton's Rings Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graph_name)
    plt.show()

if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parent.parent
    data_dir = workspace_root / "Code" / "Data"
    image_dir = workspace_root / "Reports" / "Report 2 - Newton's Rings" / "images"

    wavelength_water, wavelength_error_water, slope_water, intercept_water = analyse_and_plot(
        data_dir / 'Rings_water.csv', image_dir / 'newtons_rings_regression_water.png'
    )
    wavelength_air, wavelength_error_air, slope_air, intercept_air = analyse_and_plot(
        data_dir / 'Rings.csv', image_dir / 'newtons_rings_regression_air.png'
    )
    plot_residuals(
        data_dir / 'Rings_water.csv', slope_water, intercept_water,
        image_dir / 'residuals_water.png'
    )
    plot_residuals(
        data_dir / 'Rings.csv', slope_air, intercept_air,
        image_dir / 'residuals_air.png'
    )
    