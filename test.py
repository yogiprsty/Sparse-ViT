import torch
from tqdm import tqdm
import utils

def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')