import os
import numpy as np
from preprocessing import load_data, preprocess_data, create_mfpc
from model import PoseEstimationCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Paths
BASE_PATH = "/Users/soumya/Desktop/Study/DCS/Sensor n:w/mmWave/my_mmwave_env/bin/DB_Coursework"  # Update this to your dataset's absolute path
TRAIN_SUBJECTS = [f"S{str(i).zfill(2)}" for i in range(1, 9)]
VAL_SUBJECT = "S09"
TEST_SUBJECT = "S10"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.00001  # Further reduced learning rate
WINDOW_SIZE = 5
MAX_POINTS = 1024  # Maximum points per frame


def normalize_data(data):
    """Normalize data to have mean=0 and std=1."""
    return (data - data.mean()) / data.std()


def prepare_dataset(subjects):
    data, labels = [], []
    for subject in subjects:
        subject_path = os.path.join(BASE_PATH, subject)
        for action in os.listdir(subject_path):
            # Skip non-directory files
            if action.startswith(".") or not os.path.isdir(os.path.join(subject_path, action)):
                continue
            action_path = os.path.join(subject_path, action)
            ground_truth, mmwave_frames = load_data(action_path)
            mmwave_frames = preprocess_data(mmwave_frames)
            mfpc = create_mfpc(mmwave_frames, WINDOW_SIZE, MAX_POINTS)
            data.extend(mfpc)
            labels.extend(ground_truth[:len(mfpc)])

    # Debugging shapes and counts
    print(f"Number of data samples: {len(data)}")
    print(f"Number of labels: {len(labels)}")
    assert len(data) == len(labels), "Mismatch between data and labels!"

    return np.array(data), np.array(labels)


def initialize_weights(model):
    """Initialize model weights with Xavier initialization."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def train_model(train_loader, val_loader, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEstimationCNN(input_dim).to(device)
    initialize_weights(model)  # Apply weight initialization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # Add weight decay

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            # Add channel dimension
            data = data.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Debug loss value
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected: {loss}")
                return model

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader)}")

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.unsqueeze(1)  # Add channel dimension
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader)}")

    # Save the model after training
    torch.save(model.state_dict(), "pose_estimation_model.pth")
    print("Model saved as pose_estimation_model.pth")

    return model


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.unsqueeze(1)  # Add channel dimension
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    print(f"MAE: {mae}, RMSE: {rmse}")
    return preds, labels


if __name__ == "__main__":
    # Prepare datasets
    train_data, train_labels = prepare_dataset(TRAIN_SUBJECTS)
    val_data, val_labels = prepare_dataset([VAL_SUBJECT])
    test_data, test_labels = prepare_dataset([TEST_SUBJECT])

    # Normalize data and labels
    train_data = normalize_data(train_data)
    val_data = normalize_data(val_data)
    test_data = normalize_data(test_data)

    train_labels = normalize_data(train_labels)
    val_labels = normalize_data(val_labels)
    test_labels = normalize_data(test_labels)

    # Debug dataset ranges
    print(f"Train data range: {train_data.min()} to {train_data.max()}")
    print(f"Train labels range: {train_labels.min()} to {train_labels.max()}")

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
    val_dataset = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_labels))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Debug input_dim
    input_dim = train_data.shape[1]
    print(f"Input dimension (height of input): {input_dim}")

    # Train model
    model = train_model(train_loader, val_loader, input_dim)

    # Evaluate model
    evaluate_model(model, test_loader)




"""import os
import numpy as np
from preprocessing import load_data, preprocess_data, create_mfpc
from model import PoseEstimationCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Paths
BASE_PATH = "/Users/soumya/Desktop/Study/DCS/Sensor n:w/mmWave/my_mmwave_env/bin/DB_Coursework"  # Update this to your dataset's absolute path
TRAIN_SUBJECTS = [f"S{str(i).zfill(2)}" for i in range(1, 9)]
VAL_SUBJECT = "S09"
TEST_SUBJECT = "S10"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
#LEARNING_RATE = 0.001
LEARNING_RATE = 0.0001
WINDOW_SIZE = 5
MAX_POINTS = 1024  # Maximum points per frame


def prepare_dataset(subjects):
    data, labels = [], []
    for subject in subjects:
        subject_path = os.path.join(BASE_PATH, subject)
        for action in os.listdir(subject_path):
            # Skip non-directory files
            if action.startswith(".") or not os.path.isdir(os.path.join(subject_path, action)):
                continue
            action_path = os.path.join(subject_path, action)
            ground_truth, mmwave_frames = load_data(action_path)
            mmwave_frames = preprocess_data(mmwave_frames)
            mfpc = create_mfpc(mmwave_frames, WINDOW_SIZE, MAX_POINTS)
            data.extend(mfpc)
            labels.extend(ground_truth[:len(mfpc)])

    # Debugging shapes and counts
    print(f"Number of data samples: {len(data)}")
    print(f"Number of labels: {len(labels)}")
    assert len(data) == len(labels), "Mismatch between data and labels!"

    return np.array(data), np.array(labels)


def train_model(train_loader, val_loader, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEstimationCNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            # Add channel dimension
            data = data.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader)}")

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.unsqueeze(1)  # Add channel dimension
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader)}")

    return model


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.unsqueeze(1)  # Add channel dimension
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    print(f"MAE: {mae}, RMSE: {rmse}")
    return preds, labels


if __name__ == "__main__":
    # Prepare datasets
    train_data, train_labels = prepare_dataset(TRAIN_SUBJECTS)
    val_data, val_labels = prepare_dataset([VAL_SUBJECT])
    test_data, test_labels = prepare_dataset([TEST_SUBJECT])

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
    val_dataset = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_labels))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Debug input_dim
    input_dim = train_data.shape[1]
    print(f"Input dimension (height of input): {input_dim}")  # Debugging

    # Train model
    model = train_model(train_loader, val_loader, input_dim)

    # Evaluate model
    evaluate_model(model, test_loader)"""

