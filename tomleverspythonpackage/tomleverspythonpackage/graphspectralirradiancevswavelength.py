import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
import math

def planck_function(wavelength, T, A):
    h = 6.62607015e-34  # Planck's constant (J·s)
    c = 299792458       # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann's constant (J/K)
    spectral_irradiance = (2 * h * c**2) / (wavelength**5 * (np.exp(h * c / (k * T * wavelength)) - 1))
    r = 6.957 * 10**8
    d = 1.496 * 10**11
    conversion_factor = 1 / (10**9) # m / nm
    #return A * conversion_factor * spectral_irradiance
    # This is the actual formula.
    return math.pi * r**2 / d**2 * conversion_factor * (2 * h * c**2) / (wavelength**5 * (np.exp(h * c / (k * 5780 * wavelength)) - 1))

# Load and prepare data
df = pd.read_csv('spectral_irradiance_vs_wavelength.csv')
df = df[["date", "MIN_WAVELENGTH", "MAX_WAVELENGTH", "IRRADIANCE", "IRRADIANCE_UNCERTAINTY"]]
df["MIN_WAVELENGTH"] = df["MIN_WAVELENGTH"] * 1e-9
df["MAX_WAVELENGTH"] = df["MAX_WAVELENGTH"] * 1e-9
df = df.dropna(ignore_index = True)
mask = np.isfinite(df["MIN_WAVELENGTH"]) & np.isfinite(df["MAX_WAVELENGTH"]) & np.isfinite(df["IRRADIANCE"]) & np.isfinite(df["IRRADIANCE_UNCERTAINTY"])
df = df[mask]
df.sort_values(by = ["date"], ascending = True)
#df_sample = df.sample(n = 1000, random_state = 0)
df_sample = df[-10_000:]
column_of_average_wavelengths_in_meters = (df_sample['MIN_WAVELENGTH'] + df_sample['MAX_WAVELENGTH']) / 2
column_of_average_wavelengths_in_nanometers = column_of_average_wavelengths_in_meters * 1e9
column_of_spectral_irradiances = df_sample['IRRADIANCE']
column_of_uncertainties = df_sample['IRRADIANCE_UNCERTAINTY']

# Fit the Planck function
'''
See https://iopscience.iop.org/article/10.3847/0004-6256/152/2/41
for temperature of 5772 K.
'''
tentative_temperature_and_amplitude = [5772, 6.794e-5]  # in Kelvin and W/(m²·nm)
bounds = ([4000, 0], [7000, 1])
tuple_of_best_temperature_and_amplitude, _ = curve_fit(
    planck_function,
    column_of_average_wavelengths_in_meters,
    column_of_spectral_irradiances,
    p0 = tentative_temperature_and_amplitude,
    bounds = bounds,
    sigma = column_of_uncertainties
)
best_temperature, best_amplitude = tuple_of_best_temperature_and_amplitude
print(f"Best temperature: {best_temperature:.3f} K")
print(f"Best amplitude: {best_amplitude:.3e} W/(m²·nm)")

# Calculate predicted values and coefficient of determination (R²)
predicted_irradiances = planck_function(column_of_average_wavelengths_in_meters, best_temperature, best_amplitude)
residuals = column_of_spectral_irradiances - predicted_irradiances
rss = np.sum(residuals ** 2)
tss = np.sum((column_of_spectral_irradiances - np.mean(column_of_spectral_irradiances)) ** 2)
r_squared = 1 - (rss / tss)
print(f"Coefficient of Determination (R²): {r_squared:.3f}")

# Generate smooth curve data
linearly_spaced_wavelengths_in_meters = np.linspace(
    column_of_average_wavelengths_in_meters.min(),
    column_of_average_wavelengths_in_meters.max(),
    1000
)
linearly_spaced_wavelengths_in_nanometers = linearly_spaced_wavelengths_in_meters * 1e9
corresponding_spectral_irradiances = planck_function(
    linearly_spaced_wavelengths_in_meters,
    best_temperature,
    best_amplitude
)

# Calculate the peak of the best-fit curve
peak_index = np.argmax(corresponding_spectral_irradiances)
peak_wavelength_m = linearly_spaced_wavelengths_in_meters[peak_index]
peak_wavelength_nm = peak_wavelength_m * 1e9
peak_irradiance = corresponding_spectral_irradiances[peak_index]

print(f"Peak Wavelength: {peak_wavelength_nm:.3f} nm")
print(f"Peak Irradiance: {peak_irradiance:.3e} W/(m²·nm)")

# Define a function to add an electromagnetic spectrum display above the x-axis.
def add_em_spectrum(ax):
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks([])
    bands = [
        {"name": "UV", "min": 0, "max": 400, "color": "#FFFFFF"},
        {"name": "Violet", "min": 400, "max": 450, "color": "#9400D3"},
        {"name": "Blue", "min": 450, "max": 500, "color": "#0000FF"},
        {"name": "Cyan", "min": 500, "max": 550, "color": "#00FFFF"},
        {"name": "Green", "min": 550, "max": 580, "color": "#00FF00"},
        {"name": "Yellow", "min": 580, "max": 600, "color": "#FFFF00"},
        {"name": "Orange", "min": 600, "max": 650, "color": "#FFA500"},
        {"name": "Red", "min": 650, "max": 700, "color": "#FF0000"},
        {"name": "IR", "min": 700, "max": 2500, "color": "#FFFFFF"},
    ]
    x_min, x_max = ax.get_xlim()
    for band in bands:
        start = max(band["min"], x_min)
        end = min(band["max"], x_max)
        if start < end:
            ax_top.axvspan(
                start,
                end,
                color = band["color"],
                alpha = 0.3,
                transform = ax_top.get_xaxis_transform()
            )
            mid = (start + end) / 2
            ax_top.text(
                mid,
                1.15,
                band["name"],
                ha = "center",
                va = "bottom",
                transform = ax_top.get_xaxis_transform(),
                fontsize = "small"
            )
    ax_top.set_xlabel("")
    return ax_top

# Plot 1: Spectral Irradiance vs. Average Wavelength with R² annotation
plt.figure(figsize = (10, 6))
plt.scatter(
    column_of_average_wavelengths_in_nanometers,
    column_of_spectral_irradiances,
    s = 10,
    alpha = 0.2,
    label = "Data"
)
plt.plot(
    linearly_spaced_wavelengths_in_nanometers,
    corresponding_spectral_irradiances,
    color = "red",
    lw = 1,
    label = "Best Fit Curve"
)
# Mark the peak on the graph
plt.scatter(
    peak_wavelength_nm,
    peak_irradiance,
    color = "green",
    s = 50,
    zorder = 5,
    label = "Peak"
)
plt.annotate(
    f"Peak\n({peak_wavelength_nm:.1f} nm, {peak_irradiance:.2e})",
    xy = (peak_wavelength_nm, peak_irradiance),
    xytext = (peak_wavelength_nm + 20, peak_irradiance),
    arrowprops = dict(arrowstyle = '->', color = 'black')
)
plt.xlabel('Average Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $\left(\frac{W}{m^2\,nm}\right)$')
plt.title('Spectral Irradiance vs. Average Wavelength')
plt.grid(True)
plt.legend()

# Add R² text on the plot (placed at the upper left corner)
ax = plt.gca()
ax.text(
    0.05,
    0.95,
    f"$R^2 = {r_squared:.3f}$",
    transform = ax.transAxes,
    fontsize = 12,
    verticalalignment = 'top',
    bbox = dict(facecolor = 'white', alpha = 0.6, edgecolor = 'none')
)

add_em_spectrum(ax)
plt.show()

# Plot 2: Cumulative (Integrated) Spectral Irradiance
cumulative_irradiance = cumulative_trapezoid(
    corresponding_spectral_irradiances,
    linearly_spaced_wavelengths_in_nanometers,
    initial = 0
)
peak_index = np.argmax(cumulative_irradiance)
print("Maximum cumulative irradiance: " + str(cumulative_irradiance[peak_index]))
plt.figure(figsize = (10, 6))
plt.plot(
    linearly_spaced_wavelengths_in_nanometers,
    cumulative_irradiance,
    color = 'blue',
    lw = 1,
    label = 'Integrated Irradiance'
)
plt.xlabel('Average Wavelength (nm)')
plt.ylabel('Integrated Irradiance (W/m²)')
plt.title('Cumulative Spectral Irradiance')
plt.grid(True)
plt.legend()
ax2 = plt.gca()
add_em_spectrum(ax2)
plt.show()