# Bayesian Interferometry Repo

This repository has the Bayesia side for the Interferometry project from *Information and Decision System* Laboratory (Electrical Engineering Department, Universidad de Chile).

Here we going to implement the proposed model using pymc3 as a solver.

## Installing PYMC3
For installing correctly PYMC3 we have to pre-install the following libraries:

CUDA only works with GCC versions 4.9 or lower and Fedora 24 comes with GCC 6.1. Happily we can install 4.9.2 from the CentOS repository.

``https://drive.google.com/file/d/0B7S255p3kFXNSnR0TkJKbm5qMDQ/view?usp=sharing``

Download the linked file from there and do the following:
``bash
tar xf CentOS-SCLo-scl-el7.tar.gz
sudo cp ./etc/* /etc -rf
sudo dnf install devtoolset-3-gcc-c++
``
After that if you run:
``bash
scl enable devtoolset-3
``
bash your path will be updated and you will be using GCC 4.9.2 which you can verify with a quick ``bash gcc --version``. GCC 4.9.2 will only be used for the duration of the terminal session.

Install Tkinter
```bash
dnf install tkinter
```
```bash
pip install pymc3
```
