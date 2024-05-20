Needed Python libs

pip3 install matplotlib
pip3 install pandas



python3 NeuralNet/ny/plot_training_energies.py --datafile K_pretrain_2_2_12.csv --savefig output_K_2_2_12.pdf --ylabel "K"

python3 NeuralNet/ny/plot_training_energies_scaled.py --datafile energiesTraining_pure.csv --savefig output_energy_train_2_2_12.pdf --ylabel "E"

python3 NeuralNet/ny/plot_training_energies.py --datafile energiesTraining_pure.csv --savefig output_energy_train_2_2_12.pdf --ylabel "E"


Super-kommando for cooking pretrain data

time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 2 2 12 3 0.002 11100000 hej METROPOLIS_HASTINGS
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 2 2 16 3 0.002 11100000 hej METROPOLIS_HASTINGS

#Med bonus-kjøring
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 2 2 16 100 0.002 15800000 INTERACTION METROPOLIS_HASTINGS

100 0.002 15800000 INTERACTION METROPOLIS_HASTINGS

python3 NeuralNet/ny/plot_training_energies.py --datafile energiesTraining_pure_3.10.csv --savefig output_energy_3.10_12hidden.pdf --ylabel "E"
python3 NeuralNet/ny/plot_training_energies.py --datafile energiesTraining_pure.csv --savefig output_energy_3.10_16hidden.pdf --ylabel "E"


time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 2 2 16 100 0.002 11100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_16.csv

Chasing the magic sub 3.07
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 2 2 16 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_16.csv

