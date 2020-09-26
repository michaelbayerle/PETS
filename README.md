# Assessing Generalization using Probabilistic Ensembles for Deep Reinforcement Learning

This Code is based on the following paper: https://arxiv.org/abs/1805.12114

## Installation

Clone the code into your local system:
```
git clone https://github.com/ByMic/PETS.git
```

Optional: create a virtual environment
```
virtualenv -ppython3 env
source env/bin/activate
```

Then install the Generalization gym included in our repository:
```
cd rl-generalization
pip install -e .
```

Go back to the root level of the PETS project, and install requirements:
```
cd ..
pip install -r requirements.txt
```

## Running the Code
The main script that launches the experiments is ```mbrl.py```. The ```utils``` folder contains class files for probabilistic and deterministic networks, an ensemble class, and normalization functions. ```rl-generalization``` is a forked branch of the generalization modified OpenAi gym environments found here: ``` https://github.com/sunblaze-ucb/rl-generalization ```. The ```policies.py``` file contains the random action and MPC policies used in our experiments, and the ```data_generator.py``` holds the class that is used to roll out our policy in the selected environment to update the training dataset.

To see a list of configurable parameters, type:
```
python mbrl.py --help
```

The three different generalization tests can be run using the ```--test_type default/interpolation/extrapolation``` argument. An example of running an interpolation test on Acrobot using a probabilistic network with an ensemble size of 3, and verifying with 100 generalization test episodes, is shown below:
```
python mbrl.py --env_name Acrobot --pnn --test_type interpolation --ensemble_size 3 --test_episodes 100
```

Our final results came from averaging the scores from 5 runs each of probabalistic and deterministic networks on all generalization test types on all three environments. For example, the following demonstrates what is needed to launch a one run of all Acrobot tests:
```
python mbrl.py --env_name Acrobot --pnn --test_type default
python mbrl.py --env_name Acrobot --pnn --test_type interpolation
python mbrl.py --env_name Acrobot --pnn --test_type extrapolation

python mbrl.py --env_name Acrobot --test_type default
python mbrl.py --env_name Acrobot --test_type interpolation
python mbrl.py --env_name Acrobot --test_type extrapolation
```

## Additional Material
The results data collected from the test runs in each environment can be seen in the generalization_testing.csv under the results folder.



