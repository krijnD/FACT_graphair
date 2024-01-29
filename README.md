# DIG_FACT: FairGraph - A Fairness-focused Graph Neural Network Framework

This repository is a fork of the `fairgraph` component from the original DIG (Dive Into Graphs) project, specifically tailored for our work on fairness in graph representation learning. Our project, DIG_FACT, extends the original `fairgraph` with enhancements for hyperparameter optimization and support for a novel benchmark dataset designed to evaluate Graphair's performance on fairness metrics.
This repository was created for  ____
Please see __ for our final report and topic presentation.

## Acknowledgements
We extend our gratitude to the authors and contributors of the original DIG project for providing a solid foundation for our research. This forked repository is built upon the work presented in the paper by Ling et al. (2022) and the subsequent codebase provided by the DIVELab team.

## Project Structure
- `benchmark_dataset/`: Contains our custom benchmark dataset created to test the performance of Graphair.
- `dig/`: The core directory housing the modified fairgraph code.
- `docs/`: Documentation for the project.
- `examples/`: Example scripts and notebooks demonstrating how to use the modified fairgraph.
- `results_and_outputs/`: Output files from running the model, including performance metrics and visualizations.

## Hyperparameter Optimization
Our project includes a file `dig/fairgraph/mehtod/hpo_run.py` for running a hyperparameter search for alpha, gamma, lambda values  based on the specifications in the original paper.

## Benchmark Dataset
Within the `benchmark_dataset` directory, users will find our novel dataset. This allows for direct usage of the dataset when training and evaluating models with Graphair.
For information on the dataset please see `benchmark_dataset/README.md`.
## Getting Started
To get started with DIG_FACT, please refer to the `install_env_conda.sh` for setting up the environment and `examples/` for running your first experiments.

## Running 
We recommend for changing or editing code or running with a new dataset to check out `dig/fairgraph/mehtod/run.py`.

## License
This project code is licensed under the same terms as the original DIG project. Please see `DIG_LICENSE` for more details.

## Contributions
Contributions to DIG_FACT are welcome. Please refer to the original DIG guidelines for contributing to ensure consistency and quality.

## References


---

For any queries or further information, please open an issue in this repository or contact the maintainers.
