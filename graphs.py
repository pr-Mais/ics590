import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load the CSV data
csv_file = "loss.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Apply smoothing using Savitzky-Golay filter
smoothed_values = savgol_filter(data["Value"], window_length=5, polyorder=2)

# Plot the graph
plt.figure(figsize=(10, 6))

plt.plot(data["Step"], smoothed_values, label="Smoothed Values")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss over Time")
plt.grid(True)
plt.legend()

# Save the graph as an image file
output_file = "loss.png"  # Replace with your desired output file name
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Graph saved as {output_file}")
