# UQ-SA_foams

This repository contains the mains functions to perform foam model adjustment
to experimental data. The considered models are the Newtonian and non-Newtonian
variants implemented in **CMG-STARS**. It was originally developed in Python 3.7.4
The code is tested with the next libraries:

1. Chaospy version 4.2.5 (Forward UQ-SA)
2. PyMC3 version 3.8 (Inverse UQ)
3. SALib version 1.3.13 (SA)
4. Theano version 1.0.5 (Invese UQ)
5. lmfit version 1.0.2 (deterministic Inverse problem)

There are three demo files **input_par_Alvarez2001.dat** (here you have all the
core parameters), while **Synthetic.dat** and **Smooth.dat** (contains the two
synthetic data points used in \cite{Valdez2021}).
---

# Contact
Andres Valdez, email: arvaldez@psu.edu


---
# How to cite
If you find this library useful, cite any of the following papers:

@article{Valdez2020B,
title = {Uncertainty quantification and sensitivity analysis for relative permeability models of two-phase flow in porous media},
author = {A. R. Valdez and B M. Rocha and G. Chapiro and R. W. dos Santos},
journal={Journal of Petroleum Science and Engineering},
year={2020},
doi={https://doi.org/10.1016/j.petrol.2020.107297},
publisher={Elsevier}
}

@article{Valdez2020C,
title = {Foam assisted water-gas flow parameters: from core-flood experiment to uncertainty quantification and sensitivity analysis},
author = {A. Valdez and B. Rocha and A. Pérez-Gramatges and J. Façanha and A. de Souza and G. Chapiro and R. dos Santos },
journal={Transport in Porous Media},
year={2021},
doi={10.1007/s11242-021-01550-0},
publisher={Springer International Publishing}
}

@article{Valdez2021,
title = {Assessing uncertainties and identifiability of foam displacement models employing different objective
functions for parameter estimation},
author = {A. R. Valdez and B M. Rocha and G. Chapiro and R. W. dos Santos},
journal={Journal of Petroleum Science and Engineering},
year={2022},
publisher={Elsevier}
}
