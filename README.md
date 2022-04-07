# UQ-foams-JPSE-PETROL27715

This repository contains the mains functions to perform foam model adjustment
to experimental data. The considered models are the Newtonian and non-Newtonian
variants implemented in **CMG-STARS**. It was originally developed in Python 3.7.4
The code is tested with the next libraries:

1. PyMC3 version 3.8
2. Theano version 1.0.5

# Description

You can find here the necessary data-files to perform model calibration. We
have included three demo files. The Core sample properties are listed in
input_par_Alvarez2001.dat. The experimental records for foam quality and apparent
viscosity are available for two datasets Synthetic and Smooth.


---

# Contact
Andres Valdez, email: arvaldez@psu.edu
Bernardo Rocha, email: bernardomartinsrocha@ice.ufjf.br

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

@phdthesis{ThesisARValdez,
author = {Valdez, Andrés Ricardo},
pages = {136},
school = {PGMC-UFJF},
title = {Inverse and forward uncertainty quantification of models for foam-assisted enhanced oil recovery},
type = {Ph.D. Thesis},
year = {2021}
}

@article{Valdez2021,
title = {Assessing uncertainties and identifiability of foam displacement models employing different objective
functions for parameter estimation},
author = {A. R. Valdez and B M. Rocha and G. Chapiro and R. W. dos Santos},
journal={Journal of Petroleum Science and Engineering},
year={2022},
publisher={Elsevier}
}

