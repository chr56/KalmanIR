from typing import Union, Tuple, Sequence, Iterable, Dict, Any

DatasetName = str
BranchName = str
MetricName = str


def log_validation_metric_values(
        metric_names: Iterable[MetricName],
        current_iter: Union[int, str],
        dataset_name: str,
        metric_results: Dict[BranchName, Dict[MetricName, Any]],
        best_metric_results: Dict[MetricName, Any],
        primary_branch_name: str,
        tb_logger,
        csv_file_path: str,
):
    # Printable Log
    log_str = f'\n\t ### Validation {dataset_name} ###\n'
    log_str += printable_results(
        metric_names, metric_results, best_metric_results)
    from .logger import get_root_logger
    get_root_logger().info(log_str)
    # TensorBoard
    if tb_logger:
        for branch, branch_metric_results in metric_results.items():
            for metric, value in branch_metric_results.items():
                tb_logger.add_scalar(f'metrics_{branch}/{dataset_name}/{metric}', value, current_iter)
                if primary_branch_name == branch:
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    # CSV File
    log_validation_metric_to_csv(csv_file_path, current_iter, dataset_name, metric_results)


def printable_results(
        metric_names: Iterable[MetricName],
        metric_results: Dict[BranchName, Dict[MetricName, Any]],
        best_metric_results: Dict[MetricName, Any]
) -> str:
    from .format import TableFormatter
    formatter = TableFormatter(column_width=9, label_width=22, float_precision=4)
    formatter.header("# Metric", metric_names)

    for branch, branch_metric_results in metric_results.items():
        formatter.row_unordered(f"Output {branch:15s}", branch_metric_results)
    if best_metric_results:  # best metric is only for primary output
        best_values = []
        best_iter = []
        for metric_name in metric_names:
            best_values.append(best_metric_results[metric_name]["val"])
            best_iter.append(best_metric_results[metric_name]["iter"])
        formatter.new_line()
        formatter.row_ordered(f"Best Primary Output", best_values)
        formatter.row_ordered(f"Best Primary Iter   ", best_iter)
    formatter.new_line()
    result = formatter.result()
    return result


def log_validation_metric_to_csv(
        csv_file_path: str,
        current_iter: Union[int, str],
        dataset_name: str,
        metric_results: Dict[BranchName, Dict[MetricName, Any]],
):
    import os
    import csv
    fieldnames = ['current_iter', 'dataset', 'branch', 'metric', 'value']
    try:
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csvfile:
            # noinspection PyTypeChecker
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for branch, branch_metric_results in metric_results.items():
                for metric, raw_value in branch_metric_results.items():
                    value = str(raw_value)
                    writer.writerow({
                        'current_iter': current_iter,
                        'dataset': dataset_name,
                        'branch': branch,
                        'metric': metric,
                        'value': value,
                    })
    except Exception as e:
        from .logger import get_root_logger
        get_root_logger().error(f"Failed to log metrics to CSV: {e}")
