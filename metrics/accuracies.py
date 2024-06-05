import torch


def compute_orientability_accuracies(metrics, y_hat, y, name):
    y_hat = torch.sigmoid(y_hat)
    return [{"name": name, "value": metrics(y_hat, y)}]


def compute_name_accuracies(metrics, y_hat, y, name):
    y_hat = torch.sigmoid(y_hat)
    return [{"name": name, "value": metrics(y_hat, y)}]


def compute_betti_numbers_accuracies(metrics, y_hat, y, name):
    res = []
    for dim in range(3):
        res.append(
            {
                "name": name + f"_betti_{dim}",
                "value": metrics[dim](
                    y_hat[:, dim].round().long(), y[:, dim].long()
                ),
            }
        )
    return res
