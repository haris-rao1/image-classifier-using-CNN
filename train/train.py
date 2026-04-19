import torch
import torch.nn as nn

def train_model(model, train_loader, epochs, optimizer, device):

    # Move model to GPU if available, otherwise CPU
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Dictionary to record history
    history = {
        "train_loss": [],
        "train_acc" : []
    }
    for epoch in range(1, epochs + 1):

        model.train()  # set model to training mode

        running_loss = 0.0
        correct      = 0
        total        = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Step 1: Forward pass
            outputs = model(images)

            # Step 2: Calculate loss
            loss = criterion(outputs, labels)

            # Step 3: Zero old gradients
            optimizer.zero_grad()

            # Step 4: Backward pass
            loss.backward()

            # Step 5: Update weights
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * images.size(0)
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        # Calculate epoch stats
        epoch_loss = running_loss / total
        epoch_acc  = 100.0 * correct / total

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Loss: {epoch_loss:.4f} "
              f"Accuracy: {epoch_acc:.2f}%")

    return model, history

def save_model(model, file_name="model.pth"):
    import os
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", file_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")


def evaluate_model(model, test_loader, device):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Switch to evaluation mode
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

    test_loss = running_loss / total
    test_acc  = 100.0 * correct / total

    print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.2f}%")

    return test_acc, test_loss