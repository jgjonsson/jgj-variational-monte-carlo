import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Enable argument parsing
parser = argparse.ArgumentParser(description="Plot data from a csv file")
parser.add_argument("--datafile", help="name of the csv file", required=True)
parser.add_argument("--savefig", help="name of the figure to be saved", required=True)
parser.add_argument("--ylabel", help="label for the y-axis", required=True)

args = parser.parse_args()

# Read data from csv file and assign names to columns
df = pd.read_csv(args.datafile, sep=',', header=None, names=['Epoch nr', args.ylabel])

# Plot data
plt.plot(df['Epoch nr'], df[args.ylabel])

# Add a thin red line at y=3
plt.axhline(y=3, color='r', linewidth=0.5)
#plt.axhline(y=3, color='r', linewidth=1)

# Set y-axis limits
plt.ylim(2, 3.5)

plt.xlabel('Epoch nr')
plt.ylabel(args.ylabel)
plt.savefig(args.savefig)
