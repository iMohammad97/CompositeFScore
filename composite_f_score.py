from sklearn.metrics import precision_score, recall_score, f1_score


def event_wise_recall(y_true_events, y_pred_events):
    """
    Compute the event-wise recall.

    Parameters:
        y_true_events (list of tuples): List of ground truth anomalous events as (start, end) tuples.
        y_pred_events (list of tuples): List of predicted anomalous events as (start, end) tuples.

    Returns:
        float: Event-wise recall.
    """
    detected_events = 0

    for true_event in y_true_events:
        true_start, true_end = true_event
        for pred_event in y_pred_events:
            pred_start, pred_end = pred_event
            # Check for overlap between the true and predicted events
            if pred_end >= true_start and pred_start <= true_end:
                detected_events += 1
                break

    return detected_events / len(y_true_events) if y_true_events else 0


def pointwise_precision(y_true, y_pred):
    """
    Compute pointwise precision.

    Parameters:
        y_true (array-like): Ground truth binary labels for each time point.
        y_pred (array-like): Predicted binary labels for each time point.

    Returns:
        float: Pointwise precision.
    """
    return precision_score(y_true, y_pred)


def composite_f_score(y_true, y_pred, y_true_events, y_pred_events):
    """
    Compute the composite F-score (Fc1).

    Parameters:
        y_true (array-like): Ground truth binary labels for each time point.
        y_pred (array-like): Predicted binary labels for each time point.
        y_true_events (list of tuples): List of ground truth anomalous events as (start, end) tuples.
        y_pred_events (list of tuples): List of predicted anomalous events as (start, end) tuples.

    Returns:
        float: Composite F-score.
    """
    prt = pointwise_precision(y_true, y_pred)
    rece = event_wise_recall(y_true_events, y_pred_events)

    if prt + rece == 0:
        return 0

    fc1 = 2 * (prt * rece) / (prt + rece)
    return fc1


# Example usage
if __name__ == "__main__":
    # Ground truth binary labels for each time point
    y_true = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]

    # Predicted binary labels for each time point
    y_pred = [0, 1, 1, 1, 0, 0, 1, 0, 1, 0]

    # Ground truth anomalous events
    y_true_events = [(2, 3), (6, 8)]  # (start, end)

    # Predicted anomalous events
    y_pred_events = [(1, 3), (6, 7)]  # (start, end)

    # Calculate composite F-score
    fc1 = composite_f_score(y_true, y_pred, y_true_events, y_pred_events)
    print(f"Composite F-score (Fc1): {fc1:.2f}")

    # Calculate simple F1 score for comparison
    simple_f1 = f1_score(y_true, y_pred)
    print(f"Simple F1 Score: {simple_f1:.2f}")
