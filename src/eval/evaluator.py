import torch


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    """
    Evaluate classification accuracy of a model on a given dataloader.
    """
    model.eval()

    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc
