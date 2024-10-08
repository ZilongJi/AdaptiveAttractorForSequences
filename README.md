# Dynamical Modulation of Hippocampal Replay Sequences through Firing Rate Adaptation

Here is the code to reproduce the results in the main figures in this manuscript. 
- To reproduce figures related to the computational model, stay at the `main` branch.
- To reproduce figures related to the experimental data analysis, switch to `exp_analysis` branch and follow the instructions in the ReadMe file provided there.

## Table of Contents
1. [System Requirements](#system-requirement)
2. [Installation](#installation)
3. [Dependencies](#dependencies)
4. [Environment Setup](#environment-setup)
5. [Running the Code](#running-the-code)

## System Requirements
This code has been tested on the **linux-64** platform (Ubuntu 20.04.6 LTS) using **Visual Studio Code (1.85.1)**. Running the code on the Windows platform with Visual Studio Code should work as well.

## Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/ZilongJi/AdaptiveAttractorForSequences
cd projectname
```

## Dependencies
Below is a list of all required dependencies for the project (see environment.yml for more details):
1. **Python**: Version >= 3.8
2. **brainpy**: follow the installation [here](https://github.com/brainpy/BrainPy)
3. **numpy**
4. **scipy**
5. **matplotlib**
6. ...

## Environment Setup
To fully set up the environment using the provided `environment.yml` file, follow the steps below (tyoical installation time < 2 minutes):
```bash
conda env create -f environment.yml
conda activate your-environment-name
```

## Running the Code
Code for reproducing figures on the manuscript is located in the `Code4PaperFigures`. These files are in **Jupyter notebooks** (IPython) format, and instructions are provided in each notebook.

For example, to reproduce **Figure 2 b&c** in the paper, open `Figure2_bumpactivity_placefields_intrinsicspeed.ipynb` and excute the cells sequentially from the beginning.







