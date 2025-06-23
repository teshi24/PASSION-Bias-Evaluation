# PASSION for Dermatology
This repository contains the code to reproduce all evaluations in the paper "PASSION for Dermatology: Bridging the Diversity Gap with Pigmented Skin Images from Sub-Saharan Africa".

## Usage
Run `make` for a list of possible targets.

## Installation
Run this command for installation
`make install`

## Reproducibility of the Paper
To reproduce our experiments, we list the detailed comments needed for replicating each experiment below.
Note that our experiments were run on a DGX Workstation 1.
If less computational power is available, this would require adaptations of the configuration file.

### Experiment: Differential Diagnosis and Detecting Impetigo (Table 2 and 3)
> python -m src.evaluate_experiments --config_path configs/default.yaml --exp1 --exp2

### Experiment: Generalization across Collection Centers and Age Groups (Sec. 5, Paragraph 2 and 3)
> python -m src.evaluate_experiments --config_path configs/default.yaml --exp3 --exp4


### Fairness Evaluation as of Bachelor Thesis "Demographic Biases in Dermatology AI"
TODO: Fix docu!!!, include docu of the configs file

**Note: The code still must be cleaned and pushed to the original project, which will be done in the following days (as of 6.6.2025).**

The evaluator class got introduced to evaluate fairness. Also, the stratified splits experiment can be executed with this class.

For any experiment, the _label_ and _PASSION_split_ files must be available in the _data_ folder. They can be gathered from PASSION's repository or the zip file in the thesis.

If there is no experiment file with predictions existing, run the PASSION experiments as indicated above. Note: Currently, the code has only been tested on exp1.
The fairness evaluation is run directly for the fine-tuning process. This could be disabled by setting the _detailed_evaluation_ config to false.

If an experiment file is available in _assets/evaluation_, the evaluator can be started directly, using
> cd src/utils/
> python -m evaluator

The experiment files of the thesis can be used this way. Note that the evaluator is not yet configurable, therefore, the filenames must be enabled by uncommenting them in the evaluator class in the _main_ section.
_This is planned to be cleaned up._

To create the stratified splits, the evaluator class must be adapted, by enabling _evaluator.run_split_distribution_evaluation_ in the _main_ section and running the same script as above. Use _create_splits=True_ to create the splits. The current implementation replicates the variant with seed 32. The other version requires code adaptions.
_This is also planned to be cleaned up._

In order to recreate the full experiment, the data has to be requested via PASSION's webpage, as stated in the thesis. Consider that the currently activated model is the ResNet18 (referred as _imagenet_tiny_ in the code).



## Code and test conventions
- `black` for code style
- `isort` for import sorting
- docstring style: `sphinx`
- `pytest` for running tests

### Development installation and configurations
To set up your dev environment run:
```bash
pip install -r requirements.txt
# install pre-commit hooks
pre-commit install
```
