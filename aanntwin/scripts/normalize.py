"""This script normalizes a model"""
from statistics import mean
from statistics import median

from aanntwin.__main__ import train_and_test
from aanntwin.basic import test_model
from aanntwin.normalize import normalize_values


def main() -> None:
    """Main function"""

    model, loss_fn, test_dataloader, device = train_and_test(normalize=True)

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"Original accuracy: {accuracy}")

    for name, layer in model.store.items():
        vals = [val for tensor in layer for val in tensor.flatten().tolist()]
        print(f"{name}: {min(vals)},{max(vals)},{mean(vals)},{median(vals)}")

    model.load_named_state_dict(normalize_values(model.named_state_dict(), model.store))

    for layer in model.store.values():
        layer.clear()

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"New accuracy: {accuracy}")

    for name, layer in model.store.items():
        vals = [val for tensor in layer for val in tensor.flatten().tolist()]
        print(f"{name}: {min(vals)},{max(vals)},{mean(vals)},{median(vals)}")


if __name__ == "__main__":
    main()
