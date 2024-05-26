Needed Python libs

pip3 install matplotlib
pip3 install pandas


Sen om vi ska testa bygga for ARM
sudo apt-get install g++-arm-linux-gnueabihf
arm-linux-gnueabihf-g++ -o main main.cpp

python3 NeuralNet/ny/plot_training_energies.py --datafile K_pretrain_2_2_12.csv --savefig output_K_2_2_12.pdf --ylabel "K"

python3 NeuralNet/ny/plot_training_energies_scaled.py --datafile energiesTraining_pure.csv --savefig output_energy_train_2_2_12.pdf --ylabel "E"

python3 NeuralNet/ny/plot_training_energies.py --datafile energiesTraining_pure.csv --savefig output_energy_train_2_2_12.pdf --ylabel "E"
python3 NeuralNet/ny/plot_training_energies.py --datafile  energies_plot_pure_2_2_30_500.csv --savefig output_energy_pure_2_2_30.pdf --ylabel "E"


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

time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 2 2 20 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_20.csv

#De viktiga, men er nok samme igjen, eller nä det er 500 epoker nu
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 2 2 16 500 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_16.csv
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 2 2 20 500 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_20.csv
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 2 2 8 500 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_8.csv

python3 NeuralNet/ny/plot_training_energies.py --datafile energies_plot_pure.csv --savefig output_energy_train_2_2_20_500.pdf --ylabel "E"
python3 NeuralNet/ny/plot_training_energies.py --datafile energies_plot_pure.csv --savefig output_energy_train_2_2_20_500_.pdf --ylabel "E"
python3 NeuralNet/ny/plot_training_energies.py --datafile energies_plot_pure_2_2_12_250.csv --savefig output_energy_train_2_2_12_250.pdf --ylabel "E"


todo:
python3 NeuralNet/ny/plot_training_energies.py --datafile energies_plot_pure_2_2_30_500_0.001.csv --savefig output_energy_train_2_2_30_500_0.01.pdf --ylabel "E"



#trene nytt med learning rate 0.1, og etter omskriving reverse network til onelayer.

time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 2 2 16 3 0.002 11100000 hej METROPOLIS_HASTINGS

#Big big but sadly failed
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 3 10 20 100 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_3_10_20.csv

python3 NeuralNet/ny/plot_training_energies.py --datafile K_pretrain_3_10_20.csv --savefig output_K_3_10_20.pdf --ylabel "K"
python3 NeuralNet/ny/plot_training_energies.py --datafile K_pretrain_2_2_16.csv --savefig output_K_2_2_16.pdf --ylabel "K"
