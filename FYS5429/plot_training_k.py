import matplotlib.pyplot as plt
import pandas as pd
import argparse

#Credits to Github copilot for mostly generating this code

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

plt.xlabel('Epoch nr')
plt.ylabel(args.ylabel)
plt.savefig(args.savefig)
