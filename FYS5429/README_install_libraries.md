#Commands to install Ubuntu if running on Windows

```
wsl --install
wsl --set-default-version 2
wsl.exe --set-version Ubuntu 2
wsl.exe --install --no-distribution
wsl --install -d Ubuntu
```

#Install C++ compiler, Make and Armadillo
```
sudo apt install g++
sudo apt install make
apt-get install libarmadillo-dev

```

#Install Conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh
```
#Install Autodiff using conda
```
conda install conda-forge::autodiff
```

