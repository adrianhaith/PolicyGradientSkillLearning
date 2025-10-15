# Policy-Gradient Reinforcement Learning Models of Human Motor Skill Learning

This repository contains code to simulate human motor skill learning, as described in Haith (2025) "Policy Gradient Reinforcement Learning as a General Theory of Practice-Based Motor Skill Learning"

This repository contains three separate models:

1. Skittles Task - Simulation of learning in the skittles task developed by Muller and Sternad (Muller & Sternad, J Exp Psych, 2004)
2. Bimnaual Cursor-Control Task - Simulation of "De novo" learning a bimanual cursor-control task (Haith, et al., J Neurophys, 2022)
3. Precision Control Task - Simulation of learning precision control in the "Arc Task" (Shmuelof et al., J Neurophysiology, 2012)

In each folder, the basic scripts for running the simluations are:
'simulate_skittles_RL.py'
'simulate_bimanual_RL.py'
'simulate_arc_task_RK.py'

Scripts with '_experiment' in the filename repeat simulations multiple times to generate average learning curves.

## Setup
This code requires the following packages (listed in requirements.txt):
- numpy
- scipy
- matplotlib
- ipympl

If using anaconda (recommended) you can install these packages by typing:
`pip install -r requirements.txt'

Agents and environments are implemented in a way that is consistent with gynasium RL environments, but do not depend on the gymnasium package.

This code was developed using Python 3.11.

## Author
Adrian Haith, Baltimore, MD, USA
email: adrianhaith@gmail.com