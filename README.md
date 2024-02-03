# Optimization of synthetic oscillatory biological networks through Reinforcement Learning


This repository contains the code used ofr the experiments in the BIBM23 paper.

## Release notes

RLoscillators v0.1-BIBM23: 

* First release of RLoscillators.

## How to cite

* L. Giannantoni, A. Savino and S. Di Carlo, "Optimization of synthetic oscillatory biological networks through Reinforcement Learning," 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Istanbul, Turkiye, 2023, pp. 2624-2631, doi: 10.1109/BIBM58861.2023.10385777.

###BibTeX
```
@inproceedings{giannantoni2023optimization,
  title={Optimization of synthetic oscillatory biological networks through Reinforcement Learning},
  author={Giannantoni, Leonardo and Savino, Alessandro and Di Carlo, Stefano},
  booktitle={2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={2624--2631},
  year={2023},
  organization={IEEE}
}
```


## Experimental setup

Follow these steps to setup for reproducing the experiments provided in _Giannantoni et al., 2023_.
1) Install `Singularity` from https://docs.sylabs.io/guides/3.0/user-guide/installation.html:
    * Install `Singularity` release 3.8.3, with `Go` version 1.18.1
    * Suggestion: follow instructions provided in _Download and install singularity from a release_ section after installing `Go`
    * Install dependencies from: https://docs.sylabs.io/guides/main/admin-guide/installation.html
2) Download the container definition file
```
wget https://raw.githubusercontent.com/smilies-polito/RLoscillators/v0.1-BIBM23/RLoscillators_container.def
```
3) Build the singularity container with
```
singularity build --fakeroot RLoscillators.sif RLoscillators_container.def 
```

5) Run the singularity container with 
```
singularity run -f --writable RLoscillators.sif
```

This command will automatically run _RLoscillators_ with the default parameters.

To execute a custom experiment, run:
```
singularity run -f --writable RLoscillators.sif [arguments]
```

The accepted arguments are:
```
  -h, --help            show this help message and exit
  -m [MODEL_PATH], --model-path [MODEL_PATH]
                        model file path (default: ../data/y1_otero_rev.ant)
  -s [STEPS], --steps [STEPS]
                        number of steps (default: 32)
  -b [BATCH_SIZE], --batch-size [BATCH_SIZE]
                        batch size (default: 8)
  -l [LEARNING_RATE], --learning-rate [LEARNING_RATE]
                        learning (default: 0.001)
  -e [EPISODES], --episodes [EPISODES]
                        episodes (default: 1000)
  -v, --verbose         increase verbosity (default: False)
```