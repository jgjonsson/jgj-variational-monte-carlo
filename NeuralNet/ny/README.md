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

python3 NeuralNet/ny/NeuralNet/ny/plot_training_k.py --datafile K_pretrain_3_10_20.csv --savefig output_K_3_10_20.pdf --ylabel "K"
python3 NeuralNet/ny/NeuralNet/ny/plot_training_k.py --datafile K_pretrain_2_2_16.csv --savefig output_K_2_2_16.pdf --ylabel "K"


time ./bin/NeuralNet/ny/do_pretrain_twolayer.out 2 2 12 3 0.002 11100000 hej METROPOLIS_HASTINGS

3D forsok
fail...
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 3 5 12 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_3_5_12.csv

1D forsok
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 1 5 12 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_1_5_12.csv
python3 NeuralNet/ny/NeuralNet/ny/plot_training_k.py --datafile K_pretrainpretrain_1_5_12.csv --savefig output_K_1_5_12.pdf --ylabel "K"


python3 NeuralNet/ny/plot_training_k.py --datafile K_pretrain_1_8_12.csv --savefig output_K_1_8_12.pdf --ylabel "K"
python3 NeuralNet/ny/plot_1-K_log.py --datafile K_pretrain_1_8_12.csv --savefig output_K_1_8_12_log.pdf --ylabel "K"
python3 NeuralNet/ny/plot_1-K_log.py --datafile K_pretrain_1_5_12.csv --savefig output_K_1_5_12_log.pdf --ylabel "K"


#2-layer
time ./bin/NeuralNet/ny/do_pretrain_twolayer.out 2 2 12 3 0.002 11100000 hej METROPOLIS_HASTINGS
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained_two_layer.out 2 2 12 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_12_twolayer.csv

So this failed
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained_two_layer.out 2 2 12 250 0.002 41100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_12_twolayer.csv

Ok, does it help to retrain
time ./bin/NeuralNet/ny/do_pretrain_twolayer.out 2 2 12 3 0.002 11100000 hej METROPOLIS_HASTINGS NNparams_pretrain_2_2_12_twolayer_halfbaked.csv

python3 NeuralNet/ny/plot_training_k.py --datafile K_pretrain_2_2_12_twolayer.csv --savefig output_K_2_"_twolayer_2.pdf --ylabel "K"
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained_two_layer.out 2 2 12 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_12_twolayer.csv

time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 3 5 12 3 0.002 11100000 hej METROPOLIS_HASTINGS NNparams_pretrain_3_5_12_halfbaked.csv

python3 NeuralNet/ny/plot_1-K_log.py --datafile K_pretrain_3_5_12.csv --savefig output_K_3_5_12b_log.pdf --ylabel "K"
python3 NeuralNet/ny/plot_1-K_log.py --datafile K_pretrain_3_5_12_halfbaked.csv --savefig output_3_5_12c_log.pdf --ylabel "K"
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 3 5 12 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_3_5_12.csv
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 3 5 12 3 0.1 11100000 hej METROPOLIS_HASTINGS NNparams_pretrain_3_5_12_halfbaked2.csv

time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 3 5 30 500 0.1 11100000 hej METROPOLIS_HASTINGS
mv K_pretrain_3_5_30.csv K_pretrain_3_5_30_halfbaked.csv
mv NNparams_pretrain_3_5_30.csv NNparams_pretrain_3_5_30_halfbaked.csv
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 3 5 30 1000 0.01 11100000 hej METROPOLIS_HASTINGS NNparams_pretrain_3_5_30_halfbaked.csv

time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 3 5 30 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_3_5_30.csv

python3 NeuralNet/ny/plot_training_energies.py --datafile energies_plot_pure_3_5_30z.csv --savefig output_energy_train_3_5_30z_250.pdf --ylabel "E"


#Tilbake til
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure.out 2 2 20 3 0.1 11100000 hej METROPOLIS_HASTINGS
time ./bin/NeuralNet/ny/probe_nqs_repulsive_pure_pretrained.out 2 2 20 250 0.002 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_20_0.1_relu.csv

NNparams_pretrain_2_2_20_0.100000.csv


