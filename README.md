# Active Calibration of Reachable Sets via Approximate Pick-to-Learn
### [Paper](https://arxiv.org/)<br>

Sampada Deglurkar, Ebonye Smith, Jingqi Li, Claire Tomlin<br>
University of California, Berkeley<br> 
University of Texas at Austin

## Setup

In a conda environment, please follow the instructions to install the [Lipschitz Continuous Reachability Learning](https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning) repo.
The repo provides the gym environment for our simulated drone racing experiments, the learned
reachability value function and policy, and the implementations of the baselines.<br>

In our particular implementation, the reachability value function is modeled as a Gaussian process
(GP).
We use the GPy library to fit the value function.
You can install GPy by following the documentation [here](https://github.com/SheffieldML/GPy).

## Running Experiments

To run experiments, use the command
```
python pick_to_learn_drone/run_pick_to_learn.py
```

Optionally, add command-line arguments depending on the experiment parameters, for example
```
python pick_to_learn_drone/run_pick_to_learn.py --input_dim=3 --random_active_learning=True
```

The full list of command-line arguments is in pick_to_learn_drone/pick_to_learn_settings.py.
To print out experiment results, use
```
python pick_to_learn_drone/run_pick_to_learn.py --experiment_pickle_reader=True
```

## Citation

<!--
## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{bansal2020deepreach,
    author = {Bansal, Somil
              and Tomlin, Claire},
    title = {{DeepReach}: A Deep Learning Approach to High-Dimensional Reachability},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year={2021}
}
```
--->

<!-- 
## Contact
If you have any questions, please feel free to email the authors.
-->

