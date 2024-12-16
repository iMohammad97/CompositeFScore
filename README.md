
# Composite F1 Score Calculation Module

This repository provides a Python module for calculating the **Composite F1 Score (Fc1)**, a metric designed for evaluating time-series anomaly detection models. Unlike traditional pointwise F1 scores, this metric combines **pointwise precision** and **event-wise recall** to give a more comprehensive evaluation of anomaly detection systems.

## Features

- **Event-wise Recall Calculation**: Measures the proportion of ground truth anomalous events that are correctly detected.
- **Pointwise Precision Calculation**: Measures the accuracy of anomaly predictions at each time point.
- **Composite F1 Score**: Combines pointwise precision and event-wise recall to evaluate both point-level and event-level detection performance.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/iMohammad97/CompositeFScore.git
   ```
2. Navigate to the directory:
   ```bash
   cd CompositeFScore
   ```
3. Install dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Import the Module
```python
from composite_f1_module import composite_f_score, event_wise_recall, pointwise_precision
```

### 2. Input Formats

#### Ground Truth Events (`y_true_events`)
A list of tuples where each tuple represents a detected anomalous event:
```python
y_true_events = [(2, 5), (10, 15)]
```

#### Predicted Events (`y_pred_events`)
A list of tuples where each tuple represents a predicted anomalous event:
```python
y_pred_events = [(3, 6), (10, 12)]
```

#### Ground Truth Labels (`y_true`)
A binary array where `1` indicates an anomaly and `0` indicates normalcy:
```python
y_true = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
```

#### Predicted Labels (`y_pred`)
A binary array where `1` indicates a predicted anomaly and `0` indicates normalcy:
```python
y_pred = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
```

### 3. Example Code

```python
# Example inputs
y_true = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
y_true_events = [(2, 3), (6, 8)]
y_pred_events = [(1, 3), (6, 7)]

# Calculate Composite F1 Score
fc1 = composite_f_score(y_true, y_pred, y_true_events, y_pred_events)
print(f"Composite F1 Score (Fc1): {fc1:.2f}")
```

### 4. Output
The module will output the Composite F1 Score, providing a balanced evaluation of the anomaly detection performance.

## Contribution

We welcome contributions to improve this module! Please fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This implementation was inspired by the [paper](https://arxiv.org/abs/2109.11428) describing the Composite F1 Score for time-series anomaly detection.
