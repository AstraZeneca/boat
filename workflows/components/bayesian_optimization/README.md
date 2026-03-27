# Multi-objective Bayesian optimization

## Run MOBO

To run the code call:
```bash
poetry run src/run_bo.py
```

### Configuration of the MOBO experiment

Please refer to `configs/default_config.yaml` for a good default version of the Bayesian optimization procedure.
This document contains a detailed account of all parameters to be specified in the config.

- `experiment` (string)
Identifier/name of the experiment.

- `random_seed` (integer):
Seed for random number generation to ensure reproducibility.

- `output_dir` (string):
Directory where output files will be saved.

- `output_file` (string):
Name of the output file for storing the iterations of the genetic algorithm.

- `cluster` (string):
Identifier for the target compute cluster, e.g. `jade` or `iron`.

#### `input`
The input section defines the experimental setup, i.e., the target and the allowed mutations.

- `path_to_parental` (string, optional):
    Path to the FASTA file containing the parental sequence. You can also provide the sequence directly via the command line.

- `mutations_yaml` (string):
    Path to the YAML file defining mutations. If there are mutations for both heavy and light chain, the positions of the mutations are given with respect to the concatenated chains (heavy first).

#### `objective_functions`
Define here any number of objective functions you want to optimize. Each new entry starts with `- name:`. Options are `oasis`, and `plm`. Each option requires a particular set of parameters:

**OASIS humanness prediction**

```yaml
- name: oasis  # name of the scoring function
  description: oasis  # label that helps you identify the scoring function
```

**PLM likelihood score**

```yaml
- name: plm  # name of the scoring function
  checkpoint_path: /path/to/esm/checkpoint/
  description: plm
```

#### Bayesian Optimization Section (`bayes_opt`)
- `n_iterations` (integer):
    Number of iterations for the Bayesian optimization loop.

- `model` (string):
    The Gaussian process model used for optimization.
    **Options:** `TanimotoGP`

- `encoding` (string):
    The sequence encoding scheme used.
    **Options:** `one-hot`, `bag-of-aas`, `blosum`, `ablang2`

- `acquisition` (string):
    The acquisition function for sampling new points. `EI` and `qEI` are for single, the others for multi-objective optimization. `q` refers to batch versions (see below).
    **Options:** `EI`, `qEI`, `EHVI`, `qEHVI`, `qNEHVI`.

- `batch_size` (integer):
    Number of sequences to suggest in each batch of the Bayesian optimization loop. If >1, you need to specify a `q` acquisition function above.

#### Genetic Algorithm Section (`genetic_algorithm`)
- `initial_max_mutations` (integer):
    Maximum number of mutations allowed in the initial candidate sequences.

- `initial_population_size` (integer):
    Size of the initial population used by the genetic algorithm.

- `population_size` (integer):
    The target population size for every generation of the genetic algorithm.

- `max_rounds` (integer):
    Maximum number of rounds (generations) in the genetic algorithm process.

- `mutation_rate` (float):
    Probability of mutation occurring in each offspring.

- `crossover_rate` (float):
    Probability of performing crossover between individuals.

- `repetitions` (integer):
    Maximum number of times a sequence can be evaluated/replicated during the optimization process.

- `tournament_size` (integer):
    Number of individuals participating in tournament selection for parent selection.

- `liability_filtering` (boolean):
    Whether to filter sequences based on a liability score.

- `n_mutations` (integer):
    Specifies the upper limit of mutations that are generated in the genetic algorithm.

- `limit_mutations` (boolean):
    Whether to strictly limit the sequences to exactly n_mutations.

- `path_probability_matrix` (str):
    Path to a probability matrix as a yaml file of the type `dict[int, dict[str, float]]`. The first index represents the position within the sequence (account for concatenation of heavy and light), the second index represents the amino acid and the float value its score from any probability prior.
