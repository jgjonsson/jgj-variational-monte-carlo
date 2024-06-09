# Variational Monte Carlo solver for FYS4411 and Neural Networks for FYS5429

This repo contains code for Project 1 and Project 2 on course [FYS4411 (spring 2023)] which Gabriel Jonsson and Pavlo Panasiuk worked on, 
as well as code for FYS5429 which Gabriel Jonsson worked on.

For content on course FYS4411, see [FYS4411 README](README_FYS4411.md), and subdirectories Project-1 and Project-2 for the respective projects.

For content on course FYS5429, the code for this is mainly under directory FYS5429. See [FYS5429 README](FYS5429/README.md) for further instructions how to build and run the programs for this course.

The directories src and include contains all reusable source code components that are being used from both FYS4411 and FYS5429.
Note that even though alot of work done on neural networks, only applicable to FYS5429 is still under src because it can potentially be reused for other projects.
Also the way the Makefile is built, it limits all shared code to be under src and include (which is a bit of technical debt that would be good to sort out one day).

Furthermore some reports in pdf for are saved under reports. This to enable to cite previous work in later reports, as the reports are not published anywhere else.
