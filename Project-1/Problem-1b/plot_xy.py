
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# enable argument parsing
parser = argparse.ArgumentParser(description="Plot data from a csv file")
parser.add_argument("--datafile", help="name of the csv file")
parser.add_argument("--savefig", help="name of the figure to be saved",
                    type=str, default="temp.pdf")
args = parser.parse_args()

# Read data from csv file and assign names to columns
df = pd.read_csv(args.datafile, names=['x', 'y'], sep=' ', header=None)

# Plot data
plt.plot(df['x'], df['y'], 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(args.savefig)