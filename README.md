# RNN Knapsack Optimization

An experimental LSTM-based approach to the **0/1 knapsack problem**, evaluated against exact solutions obtained with dynamic programming.

The objective of the project is not to replace an exact optimization algorithm, but to investigate whether a recurrent neural network can learn item-selection patterns across different families and sizes of knapsack instances.

## Problem

Given a set of items, each item has:

* a weight `wᵢ`;
* a value `vᵢ`;
* a binary decision indicating whether the item is selected.

The goal is to maximize the total value of selected items without exceeding the knapsack capacity:

```text
maximize   Σ vᵢxᵢ

subject to Σ wᵢxᵢ ≤ C
           xᵢ ∈ {0, 1}
```

Exact solutions are generated using the classic dynamic programming algorithm and used as target labels for the neural network.

## Approach

The experimental pipeline consists of five stages:

1. Generate synthetic knapsack instances.
2. Obtain an optimal selection mask using dynamic programming.
3. Train an LSTM to predict the probability of selecting each item.
4. Convert the predicted probabilities into a feasible solution using greedy post-processing.
5. Compare the predicted selection with the exact solution.

The post-processing step sorts items by predicted probability and adds them while the capacity constraint remains satisfied. As a result, the neural network output always produces a feasible knapsack solution.

## Experiment design

The benchmark contains **45 experimental configurations**:

| Parameter                   | Values                                                                    |
| --------------------------- | ------------------------------------------------------------------------- |
| Instance size               | Small, Medium, Large                                                      |
| Weight–value relationship   | Uncorrelated, Weakly correlated, Strongly correlated, Inverse, Subset Sum |
| Capacity ratio `α`          | 0.25, 0.50, 0.75                                                          |
| Instances per configuration | 100                                                                       |
| Random seed                 | 42                                                                        |

Instance sizes:

| Group  | Number of items |
| ------ | --------------: |
| Small  |           20–40 |
| Medium |          60–100 |
| Large  |         150–250 |

The capacity of each instance is calculated as:

```text
C = α · Σwᵢ
```

## Input representation

Each item is represented by three features:

```text
wᵢ / C
vᵢ / max(v)
1
```

where:

* `wᵢ / C` is the normalized item weight;
* `vᵢ / max(v)` is the normalized item value;
* the constant feature represents the common capacity context.

Sequences of different lengths are padded inside each batch. A binary mask prevents padded positions from affecting the loss and evaluation metrics.

## Model

The default model consists of:

* a two-layer LSTM;
* hidden size of 256;
* dropout of 0.3;
* a linear output layer;
* sigmoid probabilities for item selection.

Training configuration:

* AdamW optimizer;
* learning rate `3e-4`;
* weight decay `5e-4`;
* cosine annealing with warm restarts;
* gradient clipping;
* early stopping;
* TensorBoard logging.

The loss is binary cross-entropy calculated only over non-padded sequence positions.

## Evaluation

The predicted probabilities are converted into a feasible binary solution using greedy packing.

The project reports:

* accuracy;
* recall;
* precision;
* average normalized value of the predicted selection.

A direct comparison between an RNN solution and the exact dynamic programming solution is also available in `solve_with_rnn.py`. The script reports the selected items, total weight, total value and optimality gap for the same problem instance.

The complete experimental summary is stored in:

```text
experiments/results/summary.csv
```

Detailed results for individual configurations are available in:

```text
experiments/results/
results/
```

## Visualizations

### Accuracy by instance-size group

![Accuracy by instance size](figures/4_4_1/accuracy_by_group.png)

### Recall by instance-size group

![Recall by instance size](figures/4_4_1/recall_by_group.png)

The results show that model quality depends substantially on the size and statistical structure of the generated knapsack instances.

## Repository structure

```text
.
├── data/                       # Generated knapsack datasets
├── experiments/
│   ├── configs/               # YAML experiment configurations
│   └── results/               # Evaluation results and summary
├── figures/                    # Aggregated visualizations
├── models/                     # Trained PyTorch model weights
├── results/                    # Training loss history
├── generate.py                 # Synthetic dataset generation
├── make_configs.py             # Generation of all 45 YAML configs
├── train.py                    # LSTM training pipeline
├── evaluation.py               # Model evaluation
├── run_experiments.py          # Complete experiment runner
├── solve_with_rnn.py           # RNN and exact DP comparison
├── analyse_4_4_1.py            # Aggregation and visualization
├── visualization.py            # Training-curve visualization
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Jysee/rnn-knapsack-optimization.git
cd rnn-knapsack-optimization
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running one experiment

Select a configuration, for example:

```text
experiments/configs/Small_Uncorrelated_a50.yaml
```

Generate the dataset:

```bash
python generate.py --config experiments/configs/Small_Uncorrelated_a50.yaml
```

Train the model:

```bash
python train.py --config experiments/configs/Small_Uncorrelated_a50.yaml
```

Evaluate the trained model:

```bash
python evaluation.py \
  --config experiments/configs/Small_Uncorrelated_a50.yaml \
  --outfile experiments/results/Small_Uncorrelated_a50_eval.csv
```

On Windows PowerShell, the evaluation command can be written on one line:

```powershell
python evaluation.py --config experiments/configs/Small_Uncorrelated_a50.yaml --outfile experiments/results/Small_Uncorrelated_a50_eval.csv
```

## Running the complete benchmark

Generate configurations if necessary:

```bash
python make_configs.py
```

Run all 45 experiments:

```bash
python run_experiments.py
```

This command regenerates the datasets, trains a separate model for every configuration, evaluates the models and creates a combined CSV summary.

## Comparing RNN and dynamic programming

A trained model can be compared with the exact solution on a single instance:

```bash
python solve_with_rnn.py
```

The model path, item weights, item values and capacity can be changed at the beginning of the script.

Example output:

```text
==== RNN solution ====
Selected indices: [...]
Total weight: ... / ...
Total value: ...

==== DP optimal solution ====
Selected indices: [...]
Total weight: ... / ...
Total value: ...

Optimality gap: ...%
```

## Limitations

* The model is trained on synthetic data.
* An LSTM is sensitive to the order in which items are presented.
* Several different item selections may have the same optimal value, while the model is trained against one particular optimal mask.
* Classification metrics do not fully represent optimization quality.
* Dynamic programming remains preferable when an exact solution is computationally feasible.
* Generalization outside the generated instance distributions has not been established.

A useful extension would be to calculate the average optimality gap and inference time over the held-out test set and compare the model with simple heuristic baselines.

## Technologies

* Python
* PyTorch
* NumPy
* pandas
* scikit-learn
* Matplotlib
* Seaborn
* TensorBoard
* YAML
