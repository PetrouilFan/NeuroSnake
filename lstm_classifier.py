import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration Constants
RECORDINGS_DIR = "recordings"
MODELS_DIR = "models"
SEQUENCE_LENGTH = 200  # Number of time steps in each sequence
HIDDEN_SIZE = 128  # Number of features in LSTM hidden state
NUM_LAYERS = 2  # Number of stacked LSTM layers
BATCH_SIZE = 256
LEARNING_RATE = 0.002
EPOCHS = 30
TEST_SPLIT = 0.2  # Percentage of data for testing
VAL_SPLIT = 0.2  # Percentage of training data for validation
DROPOUT = 0.2  # Dropout probability
BIDIRECTIONAL = True  # Whether to use bidirectional LSTM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42  # For reproducibility
SAVE_MODEL = True  # Whether to save the trained model
EARLY_STOPPING = True  # Whether to use early stopping
PATIENCE = 7  # Number of epochs with no improvement after which training will stop
INPUT_SIZE = 3  # Number of features (attention, meditation, raw_eeg)
OUTPUT_SIZE = 4  # Number of classes (UP, DOWN, LEFT, RIGHT)

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_and_preprocess_data():
    """
    Load and preprocess data from all CSV files in the recordings directory.
    
    Returns:
    --------
    X : numpy.ndarray
        Input features normalized
    y : numpy.ndarray
        Target labels as one-hot encoded vectors
    scaler : sklearn.preprocessing.RobustScaler
        Fitted scaler for normalizing new data
    """
    print("Loading and preprocessing data...")
    
    # Get all CSV files in the recordings directory
    csv_files = glob.glob(os.path.join(RECORDINGS_DIR, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {RECORDINGS_DIR}")
    
    # Initialize lists to store data
    all_data = []
    
    # Load each CSV file
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    # Concatenate all data into a single dataframe
    data = pd.concat(all_data, ignore_index=True)
    
    # Display some basic information
    print(f"Total records: {len(data)}")
    
    # Convert direction to numerical labels
    direction_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
    data['direction_label'] = data['direction'].map(direction_mapping)
    
    # Count occurrences of each direction
    direction_counts = data['direction'].value_counts()
    print("Direction counts:")
    print(direction_counts)
    
    # Print statistics about the features
    print("\nData statistics for features:")
    for feature in ['attention', 'meditation', 'raw_eeg']:
        print(f"\n{feature.upper()}:")
        print(f"Min: {data[feature].min()}")
        print(f"Max: {data[feature].max()}")
        print(f"Mean: {data[feature].mean()}")
        print(f"Median: {data[feature].median()}")
        print(f"Standard Deviation: {data[feature].std()}")
        print(f"25th Percentile: {data[feature].quantile(0.25)}")
        print(f"75th Percentile: {data[feature].quantile(0.75)}")
    
    # Extract features and target
    X = data[['attention', 'meditation', 'raw_eeg']].values
    y = data['direction_label'].values
    
    # Normalize features with RobustScaler
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Print statistics after normalization
    print("\nData statistics after robust normalization:")
    feature_names = ['attention', 'meditation', 'raw_eeg']
    for i in range(X.shape[1]):
        print(f"\n{feature_names[i].upper()}:")
        print(f"Min: {X[:, i].min()}")
        print(f"Max: {X[:, i].max()}")
        print(f"Mean: {X[:, i].mean()}")
        print(f"Standard Deviation: {X[:, i].std()}")
    
    # One-hot encode the target
    y_one_hot = np.zeros((len(y), OUTPUT_SIZE))
    for i, label in enumerate(y):
        y_one_hot[i, label] = 1
    
    return X, y_one_hot, scaler

class EEGSequenceDataset(Dataset):
    """
    Dataset for creating sequences of EEG data for LSTM input.
    """
    
    def __init__(self, X, y, seq_length=SEQUENCE_LENGTH):
        """
        Initialize the dataset with features and targets.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target labels as one-hot encoded vectors
        seq_length : int
            Length of sequences to create
        """
        self.X = X
        self.y = y
        self.seq_length = seq_length
        
    def __len__(self):
        # Number of possible sequences with the given sequence length
        return max(0, len(self.X) - self.seq_length)
    
    def __getitem__(self, idx):
        # Get a sequence of features and the corresponding target
        # We use the target of the last timestep in the sequence
        X_seq = self.X[idx:idx+self.seq_length]
        y_seq = self.y[idx+self.seq_length-1]
        
        # Convert to tensors
        X_seq = torch.FloatTensor(X_seq)
        y_seq = torch.FloatTensor(y_seq)
        
        return X_seq, y_seq

class EEGLSTMClassifier(nn.Module):
    """
    LSTM classifier for EEG data.
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, 
                 output_size=OUTPUT_SIZE, dropout=DROPOUT, bidirectional=BIDIRECTIONAL):
        """
        Initialize the LSTM classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of expected features in the input
        hidden_size : int
            Number of features in the hidden state
        num_layers : int
            Number of recurrent layers (stacked LSTM)
        output_size : int
            Number of classes to predict
        dropout : float
            Dropout probability for LSTM layers
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(EEGLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        # Fully connected output layer
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(DEVICE)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size * num_directions)
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output from the last time step
        if self.bidirectional:
            # If bidirectional, concatenate the last output from forward and backward passes
            out = out[:, -1, :]
        else:
            # If unidirectional, just take the last output
            out = out[:, -1, :]
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out
    
    def predict(self, x):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        torch.Tensor
            Predicted class probabilities
        """
        # Set the model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(x)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        return probabilities

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    """
    Train the model.
    
    Parameters:
    -----------
    model : EEGLSTMClassifier
        The model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    epochs : int
        Number of epochs to train for
        
    Returns:
    --------
    model : EEGLSTMClassifier
        Trained model
    train_losses : list
        Training loss history
    val_losses : list
        Validation loss history
    """
    # Initialize lists to track metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Training on {DEVICE}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            train_loss += loss.item()
            
        # Average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                _, target_indices = torch.max(target, 1)
                total += target.size(0)
                correct += (predicted == target_indices).sum().item()
        
        # Average validation loss and accuracy for the epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Training Loss: {avg_train_loss:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f} | "
              f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping check
        if EARLY_STOPPING:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model so far
                if SAVE_MODEL:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"eeg_lstm_best_{timestamp}.pth"))
                    print(f"Best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
                print(f"EarlyStopping counter: {patience_counter} out of {PATIENCE}")
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODELS_DIR, 'training_loss.png'))
    plt.close()
    
    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODELS_DIR, 'validation_accuracy.png'))
    plt.close()
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set.
    
    Parameters:
    -----------
    model : EEGLSTMClassifier
        The trained model
    test_loader : DataLoader
        DataLoader for test data
        
    Returns:
    --------
    accuracy : float
        Test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Forward pass
            output = model(data)
            
            # Get predicted class
            _, predicted = torch.max(output, 1)
            _, target_indices = torch.max(target, 1)
            
            # Update counters
            total += target.size(0)
            correct += (predicted == target_indices).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target_indices.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, np.array(all_predictions), np.array(all_targets)

def main():
    """
    Main function to run the model training and evaluation.
    """
    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data()
    
    # Split data into train, validation, and test sets
    # First split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    
    # Then split train+val into train and val sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create datasets
    train_dataset = EEGSequenceDataset(X_train, y_train)
    val_dataset = EEGSequenceDataset(X_val, y_val)
    test_dataset = EEGSequenceDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = EEGLSTMClassifier().to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS
    )
    
    # Evaluate model
    accuracy, predictions, targets = evaluate_model(model, test_loader)
    
    # Save the trained model
    if SAVE_MODEL:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODELS_DIR, f"eeg_lstm_final_{timestamp}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Also save the scaler for future use
        import joblib
        scaler_path = os.path.join(MODELS_DIR, f"eeg_scaler_{timestamp}.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
    print("Training and evaluation complete.")

    # Example of initializing the real-time classifier
    # (Uncomment and modify with the correct model path to use)
    # rt_classifier = RealTimeEEGClassifier(model_path, scaler)
    # predicted_direction, probs = rt_classifier.predict(50, 50, 100)
    # print(f"Predicted direction: {predicted_direction}")
    # print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()
