import os
import glob
import time
import joblib
import torch
import pygame  # New import for event processing

from mindwave_connection import MindwaveConnection  # For EEG headset connection
from snake_game import SnakeGame  # Assuming a SnakeGame implementation exists

# Locate the latest saved model and scaler in models/
MODELS_DIR = os.path.join("models")
model_files = glob.glob(os.path.join(MODELS_DIR, "eeg_lstm_final_*.pth"))
scaler_files = glob.glob(os.path.join(MODELS_DIR, "eeg_scaler_*.pkl"))

if not model_files or not scaler_files:
    raise ValueError("No model or scaler found in the models directory.")

latest_model = max(model_files, key=os.path.getctime)
latest_scaler = max(scaler_files, key=os.path.getctime)
print(f"Loading model: {latest_model}")
print(f"Loading scaler: {latest_scaler}")

# Load scaler
scaler = joblib.load(latest_scaler)

# Instantiate the real-time classifier with the latest model and scaler
rt_classifier = RealTimeEEGClassifier(model_path=latest_model, scaler=scaler)

# Initialize EEG connection (using same COM port as in data_recorder)
COM_PORT = "COM4"  # Modify if necessary
headset_connection = MindwaveConnection(COM_PORT)
if not headset_connection.connect():
    raise ConnectionError("Failed to connect to the EEG headset.")

class RealTimeEEGClassifier:
    """
    Class for real-time classification of EEG data.
    """
    
    def __init__(self, model_path, scaler, seq_length=SEQUENCE_LENGTH):
        """
        Initialize the real-time classifier.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model
        scaler : sklearn.preprocessing.StandardScaler
            Fitted scaler for normalizing data
        seq_length : int
            Sequence length expected by the model
        """
        # Load the model
        self.model = EEGLSTMClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        
        # Store scaler and sequence length
        self.scaler = scaler
        self.seq_length = seq_length
        
        # Initialize buffer for storing sequence data
        self.data_buffer = []
        
        # Direction mapping
        self.direction_mapping = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        
    def preprocess_sample(self, attention, meditation, raw_eeg):
        """
        Preprocess a single EEG sample.
        
        Parameters:
        -----------
        attention : int or float
            Attention value
        meditation : int or float
            Meditation value
        raw_eeg : int or float
            Raw EEG value
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed sample
        """
        # Create a sample
        sample = np.array([[attention, meditation, raw_eeg]])
        
        # Normalize the sample
        sample_normalized = self.scaler.transform(sample)
        
        return sample_normalized[0]
    
    def update_buffer(self, sample):
        """
        Add a new sample to the buffer and maintain the buffer size.
        
        Parameters:
        -----------
        sample : numpy.ndarray
            New sample to add to the buffer
        """
        self.data_buffer.append(sample)
        
        # Keep only the last seq_length samples
        if len(self.data_buffer) > self.seq_length:
            self.data_buffer.pop(0)
    
    def predict(self, attention, meditation, raw_eeg):
        """
        Make a prediction for the current EEG state.
        
        Parameters:
        -----------
        attention : int or float
            Attention value
        meditation : int or float
            Meditation value
        raw_eeg : int or float
            Raw EEG value
            
        Returns:
        --------
        str
            Predicted direction
        numpy.ndarray
            Class probabilities
        """
        # Preprocess the sample
        sample = self.preprocess_sample(attention, meditation, raw_eeg)
        
        # Add to buffer
        self.update_buffer(sample)
        
        # Check if we have enough data
        if len(self.data_buffer) < self.seq_length:
            # Not enough data for prediction yet
            return None, None
        
        # Convert buffer to tensor
        X = np.array(self.data_buffer)
        X = torch.FloatTensor(X).unsqueeze(0).to(DEVICE)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = self.model(X)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get the predicted class
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        
        # Convert to direction
        predicted_direction = self.direction_mapping[predicted_class]
        
        return predicted_direction, probabilities.cpu().numpy()[0]
    
    def reset_buffer(self):
        """Reset the data buffer."""
        self.data_buffer = []

# Initialize the game instance (assuming a SnakeGame with a similar API)
game = SnakeGame()

# Create a pygame clock for stable frame rate
clock = pygame.time.Clock()

# Start the headset streaming if needed
if not hasattr(headset_connection, 'running') or not headset_connection.running:
    headset_connection.start()

# Main loop: capture EEG, predict action, and update the game frame
try:
    running = True
    while running:
        # Process UI events to keep the game window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # We'll handle keyboard inputs through the next_frame method
        
        # Sample EEG data from the headset
        attention = headset_connection.attention
        meditation = headset_connection.meditation
        raw_eeg = headset_connection.raw_value
        
        # Default control flags
        up = down = left = right = False
        
        # Get predicted direction
        predicted_direction, probabilities = rt_classifier.predict(attention, meditation, raw_eeg)
        
        if predicted_direction is not None:
            print(f"Predicted direction: {predicted_direction} | Probabilities: {probabilities}")
            # Map prediction to game controls
            if predicted_direction.upper() == "UP":
                up = True
            elif predicted_direction.upper() == "DOWN":
                down = True
            elif predicted_direction.upper() == "LEFT":
                left = True
            elif predicted_direction.upper() == "RIGHT":
                right = True
        
        # Update the game with predicted controls
        game_over = game.next_frame(up, down, left, right)
        
        # Draw the game
        game.draw()
        
        if game_over:
            print("Game over! Resetting...")
            time.sleep(1)  # Brief pause to show game over
            game.reset_game()
        
        # Maintain consistent frame rate
        clock.tick(game.FRAME_RATE)
        
except KeyboardInterrupt:
    print("Exiting brain controller.")

# Clean up resources
pygame.quit()
headset_connection.stop()
try:
    game.cleanup()
except AttributeError:
    # If cleanup method doesn't exist, we're already done
    pass
