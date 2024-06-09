# Variational Monte Carlo solver for FYS4411

This repo contains the code for Project 1 and Project 2 on course [FYS4411 (spring 2023)](https://github.com/CompPhysics/ComputationalPhysics2).
It's based on a fork of the initial code repo provided in that course, and further developed by Gabriel Jonsson and Pavlo Panasiuk.

As we reuse much of the code between the two projects, they are both located in the same source tree folder, with new files for project 2 mereley added.
Latest on master will thus at the time of this writing show the most recent status of Project 2.
If you want to go back to the state of Project 1 completion, run:
```
git checkout Project1
```

### Compiling the project using Make

The Makefile is set up, such that each program is built separately.
Programs for various sub-tasks in the project are organized in subfolders under Project1, while common code files are placed under src/ and include/

In general the build command is like follows where app_name_no_ext.cpp is some source code file containing main function. (Simple running make without parameter does not produce executables in the current setting).

In a Linux/Mac terminal this can be done by the following commands
```bash
make app=path_to_app/app_name_no_ext
```
and this can be executed with
```bash
./bin/path_to_app/app_name_no_ext.out
```

For one concrete example, the build command for main executable for sub-task 1g is:
```
make app=Project-1/Problem-1g/repulsive_hamiltonian
```

See README.md file in respective subfolder of Project1 for instructions how to build and run, with parameters relevant for the project report.

#### Cleaning the directory
Run `make clean` in the top-directory to clean up generated object files. For cleaning program binaries as well, specify the `app`-argument, e.g., `make clean app=path_to_app/app_name_no_ext`.

