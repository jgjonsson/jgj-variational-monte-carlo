Credits to Marius Jonsson for this script for calculating standard error using blocking method.

Script taken from:
https://github.com/computative/block/blob/master/python/tictoc.py
Only changes made is renaming tictoc.py to blocking.py and changing the hardcided data filename in the script to energies.csv.

Marius Jonsson's article:
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.043304

Script expects a file in current location named energies.csv containing one floating point number per row. 
Running script from directory where data file is: 
python3 <path to dir>\blocking.py
