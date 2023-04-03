
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
df = pd.read_csv(args.datafile, sep=' ', header=None)

# Plot data
plt.plot(df[0], df[1], 'o')
plt.xlabel('alpha par.')
plt.ylabel('<E>, a.u.')
plt.savefig(args.savefig)