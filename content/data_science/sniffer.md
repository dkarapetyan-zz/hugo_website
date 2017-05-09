+++
title="Sniffer"
image="/images/sniffer.jpg"
description="An occupancy sniffer for small establishments (using raspberry pi)"
+++

# Synopsis

This is an occupancy sniffer for small establishments. 
It uses a wifi dongle, connected to a rasperry pi, to collect data 
from wifi signals in the surrounding area. Then, using a proprietary algorithm to filter and process the data, it computes the occupancy of the establishment at periodic intervals.  

# Installation

* [Install the Python 2.7 version of 
Anaconda 4.0.0 (64-bit)](https://www.continuum.io/downloads).
* Download the desired release of Sniffer 
to your local hard drive. If you have a local copy of git, this can be
done by running `git clone https://github.com/dkarapetyan/sniffer` in a unix
shell.
* From a unix shell, run `pip -e install $PROJECT_ROOT`. After installation,
do not move the `$PROJECT_ROOT` directory, as this will break the
installation.

# Execution

* Once installation is successful, execute `run_sniffer` in a bash shell 
  (it is automatically added to your `PATH` environment variable 
  by the installation process). This is the entry point for Sniffer.
  suite.

# Options and Features

* For options and features, please execute `run_sniffer -h`
  in your shell. 
 
 
# Contributors

* [David Karapetyan](mailto:david.karapetyan@gmail.com)

# License

* Proprietary
