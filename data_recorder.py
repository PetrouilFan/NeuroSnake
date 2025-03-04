import numpy as np
import pandas as pd
import time
import os
import serial
import threading
from datetime import datetime
import pygame
import csv
from mindwave_connection import MindwaveConnection

# Configuration Constants
RECORDING_DIR = "recordings"
DATA_DIR = "data"
EEG_UPDATE_RATE = 10  # Hz, rate at which to sample EEG data
DEFAULT_COM_PORT = "COM4"  # Default serial port for Mindwave
DEFAULT_RECORD_DURATION = 60  # Default recording duration in seconds
WINDOW_WIDTH = 800  # Match with snake_game.py
WINDOW_HEIGHT = 600  # Match with snake_game.py

# Signal quality thresholds
GOOD_SIGNAL_THRESHOLD = 50
MEDIUM_SIGNAL_THRESHOLD = 150


class DataRecorder:
    """
    Records EEG data along with corresponding game actions for the NeuroSnake game.
    Used to create datasets for training the AI model.
    """
    
    def __init__(self, game=None, port=DEFAULT_COM_PORT, record_duration=DEFAULT_RECORD_DURATION, data_dir=DATA_DIR):
        """
        Initialize the data recorder.
        
        Parameters:
        -----------
        game : The SnakeGame instance
        port : COM port where Mindwave is connected
        record_duration : Recording duration in seconds
        data_dir : str
            Directory where data will be saved
        """
        # Ensure pygame is initialized if we're not using an existing game
        self.pygame_initialized = pygame.get_init()
        if not self.pygame_initialized and game is None:
            pygame.init()
            self.pygame_initialized = True
        
        self.game = game
        self.port = port
        self.record_duration = record_duration
        self.data_dir = data_dir
        self.recording = False
        # Simplified EEG data structure with direction
        self.eeg_data = {
            'timestamp': [],
            'attention': [],
            'meditation': [],
            'raw_eeg': [],
            'direction': []  # Added direction to EEG data
        }
        self.recordings = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_action = None  # Track the last action
        
        # Create data directories if they don't exist
        for directory in [data_dir, RECORDING_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
        self.filename = os.path.join(data_dir, f"eeg_gameplay_data_{self.session_id}.csv")
        
        # Only initialize the font if pygame is ready
        if self.pygame_initialized or game is not None:
            self.font = pygame.font.SysFont(None, 36)
        else:
            self.font = None
            print("Warning: pygame not initialized, text display will be disabled")
        
        # Create a screen for standalone mode (when no game is provided)
        self.standalone_screen = None
        if game is None and self.pygame_initialized:
            self.standalone_screen = pygame.display.set_mode((400, 200))
            pygame.display.set_caption("EEG Recorder")
        
        # Initialize the MindWave connection
        self.headset = None
        self.connection = MindwaveConnection(port)
        self.connected = self.connection.connect()

    def start_recording(self):
        """Start recording EEG and direction data"""
        if not self.connected:
            print("Cannot start recording: Mindwave device not connected")
            return False
        
        # Start the connection if not already running
        if not hasattr(self.connection, 'running') or not self.connection.running:
            if not self.connection.start():
                return False
        
        self.recording = True
        
        # Reset EEG data
        for key in self.eeg_data:
            self.eeg_data[key] = []
        
        self.record_start_time = time.time()
        print("Recording started")
        return True
    
    def stop_recording(self):
        """Stop recording and save data"""
        if not self.recording:
            return
        
        self.recording = False
        
        # Save recorded data
        self.save_data()
        print("Recording stopped and data saved")
    
    def record_game_state(self):
        """Record current direction with EEG data"""
        if not self.recording:
            return
        
        # Only record the direction from game state if game exists
        if self.game:
            direction = self.game.get_state().get('direction', 'RIGHT')
            self._record_eeg_data(direction)
        else:
            # If no game, just record EEG data with last known direction
            self._record_eeg_data(self.last_action or 'RIGHT')
    
    def _record_eeg_data(self, direction='RIGHT'):
        """Record current EEG data with timestamp and direction"""
        if not self.recording or not self.connected:
            return
            
        timestamp = time.time()
        self.eeg_data['timestamp'].append(timestamp)
        self.eeg_data['attention'].append(self.connection.attention)
        self.eeg_data['meditation'].append(self.connection.meditation)
        self.eeg_data['raw_eeg'].append(self.connection.raw_value)
        self.eeg_data['direction'].append(direction)
    
    def save_data(self):
        """Save recorded EEG data with direction to CSV file"""
        if not any(len(values) > 0 for values in self.eeg_data.values()):
            print("No EEG data collected to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eeg_file = os.path.join(RECORDING_DIR, f"eeg_data_{timestamp}.csv")
        
        # Get all timestamps to ensure data alignment
        all_timestamps = set()
        for ts in self.eeg_data['timestamp']:
            all_timestamps.add(ts)
        
        all_timestamps = sorted(all_timestamps)
        
        # Create a list of dictionaries for each timestamp
        rows = []
        for ts in all_timestamps:
            idx = self.eeg_data['timestamp'].index(ts) if ts in self.eeg_data['timestamp'] else -1
            if idx >= 0:
                row = {'timestamp': ts}
                for key, values in self.eeg_data.items():
                    if key != 'timestamp' and idx < len(values):
                        row[key] = values[idx]
                rows.append(row)
        
        # Write the data to CSV
        if rows:
            with open(eeg_file, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'attention', 'meditation', 'raw_eeg', 'direction']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            
            print(f"EEG data saved to {eeg_file}")
    
    def draw_recording_status(self):
        """Draw recording status on the game screen"""
        # Check if pygame is initialized and we have a font
        if not self.font:
            return
            
        # Use either the game screen or our standalone screen
        screen = self.game.screen if self.game else self.standalone_screen
        if not screen:
            return
            
        if self.recording:
            # Draw blinking red recording indicator
            if int(time.time() * 2) % 2 == 0:
                indicator_color = (255, 0, 0)  # Red
            else:
                indicator_color = (200, 0, 0)  # Darker red
                
            pygame.draw.circle(screen, indicator_color, (WINDOW_WIDTH - 20 if self.game else 380, 20), 10)
            
            # Draw recording time
            elapsed = int(time.time() - self.record_start_time)
            remaining = max(0, self.record_duration - elapsed)
            time_text = self.font.render(f"Recording: {remaining}s", True, (255, 255, 255))
            screen.blit(time_text, (WINDOW_WIDTH - 160 if self.game else 220, 10))
            
            # Draw connection quality indicator
            if self.connected:
                quality = self.connection.get_signal_quality()
                quality_color = (0, 255, 0) if quality < GOOD_SIGNAL_THRESHOLD else (255, 255, 0) if quality < MEDIUM_SIGNAL_THRESHOLD else (255, 0, 0)
                quality_text = self.font.render(f"Signal: {quality}", True, quality_color)
                screen.blit(quality_text, (WINDOW_WIDTH - 100 if self.game else 160, 40))
                pygame.draw.circle(screen, quality_color, (WINDOW_WIDTH - 40 if self.game else 380, 40), 10)
    
    def run_recording_session(self):
        """Run a game session with EEG recording"""
        # Add window decoration to indicate recording mode
        pygame.display.set_caption('NeuroSnake - EEG Recording Mode')
        
        print("Starting recording session. Press SPACE to start/stop recording.")
        
        self.record_start_time = 0
        last_space_press = 0
        
        # Main game loop
        while True:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.recording:
                        self.stop_recording()
                    # Safely disconnect headset
                    self.connection.stop()
                    pygame.quit()
                    return
                    
                if event.type == pygame.KEYDOWN:
                    # Start/stop recording with space key (with debounce)
                    if event.key == pygame.K_SPACE and time.time() - last_space_press > 1.0:
                        last_space_press = time.time()
                        if not self.recording:
                            if self.start_recording():
                                self.record_start_time = time.time()
                        else:
                            self.stop_recording()
            
            # Get keyboard input for game
            keys = pygame.key.get_pressed()
            up = keys[self.game.KEY_UP]
            down = keys[self.game.KEY_DOWN]
            left = keys[self.game.KEY_LEFT]
            right = keys[self.game.KEY_RIGHT]
            
            # Process game frame
            game_over = self.game.next_frame(up, down, left, right)
            
            # Record game state if recording
            if self.recording:
                self.record_game_state()
                
                # Auto-stop recording after duration
                if time.time() - self.record_start_time >= self.record_duration:
                    self.stop_recording()
            
            # Check if game over
            if game_over:
                if self.recording:
                    self.stop_recording()
                
                font = pygame.font.SysFont(None, 72)
                game_over_text = font.render("Game Over", True, (255, 255, 255))
                self.game.screen.blit(game_over_text, (WINDOW_WIDTH//2 - game_over_text.get_width()//2, 
                                           WINDOW_HEIGHT//2 - game_over_text.get_height()//2))
                pygame.display.update()
                pygame.time.wait(2000)
                self.game.reset_game()
            
            # Draw game
            self.game.draw()
            
            # Draw recording status if device is connected
            if self.connected:
                self.draw_recording_status()
            
            pygame.display.update()
            self.game.clock.tick(self.game.FRAME_RATE)
    
    def run_standalone(self):
        """Run a standalone recording session without the game"""
        if not self.pygame_initialized or not self.standalone_screen:
            print("Cannot run standalone mode: pygame not initialized")
            return
            
        pygame.display.set_caption('NeuroSnake - EEG Recorder Standalone')
        
        print("Starting standalone recording session. Press SPACE to start/stop recording, ESC to exit.")
        
        self.record_start_time = 0
        last_space_press = 0
        running = True
        
        # Main loop
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.KEYDOWN:
                    # Exit on ESC
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    # Start/stop recording with space key (with debounce)
                    if event.key == pygame.K_SPACE and time.time() - last_space_press > 1.0:
                        last_space_press = time.time()
                        if not self.recording:
                            if self.start_recording():
                                self.record_start_time = time.time()
                        else:
                            self.stop_recording()
            
            # Clear the screen
            self.standalone_screen.fill((0, 0, 0))
            
            # Display EEG data if connected
            if self.connected:
                y_pos = 70
                texts = [
                    f"Attention: {self.connection.attention}",
                    f"Meditation: {self.connection.meditation}",
                    f"Signal: {self.connection.get_signal_quality()}",
                    f"Blink: {self.connection.blink}"
                ]
                
                for text in texts:
                    rendered = self.font.render(text, True, (255, 255, 255))
                    self.standalone_screen.blit(rendered, (20, y_pos))
                    y_pos += 30
            else:
                # If disconnected, try to reconnect
                status_text = self.font.render("Disconnected. Attempting to reconnect...", True, (255, 100, 100))
                self.standalone_screen.blit(status_text, (20, 70))
                
                # Try reconnect every 3 seconds
                if int(time.time()) % 3 == 0:
                    if not self.connected:
                        self.connected = self.connection.connect()
            
            # If recording, auto-stop after duration
            if self.recording and time.time() - self.record_start_time >= self.record_duration:
                self.stop_recording()
            
            # Draw recording status
            self.draw_recording_status()
            
            # Update the display
            pygame.display.update()
            
            # Control the frame rate
            pygame.time.delay(100)  # 10 fps
            
        # Clean up
        if self.recording:
            self.stop_recording()
        
        # Safely disconnect headset
        self.connection.stop()
        
        # Only quit pygame if we initialized it
        if self.pygame_initialized and not self.game:
            pygame.quit()
    
    def record(self, action=None, game_state=None):
        """
        Record a single data point with the current EEG data and the corresponding direction.
        If no action is provided, uses the last recorded action.
        
        Parameters:
        -----------
        action : str, optional
            Game direction ('UP', 'DOWN', 'LEFT', 'RIGHT'). If None, uses last action.
        game_state : dict, optional
            Not used anymore except for extracting direction if present
        """
        if not self.connected:
            return
            
        timestamp = time.time()
        
        # If an action is provided, update the last_action
        if action is not None:
            self.last_action = action
        # If game_state has direction info, use that
        elif game_state and 'direction' in game_state:
            self.last_action = game_state['direction']
        # If no action is provided and we have no last action yet, set a default
        elif self.last_action is None:
            self.last_action = 'RIGHT'  # Default starting direction in most Snake games
            
        # Create a simplified data point with only required fields
        data_point = {
            'timestamp': timestamp,
            'attention': self.connection.attention,
            'meditation': self.connection.meditation,
            'raw_eeg': self.connection.raw_value,
            'direction': self.last_action
        }
        
        self.recordings.append(data_point)
    
    def save(self):
        """
        Save the recorded data to a CSV file.
        """
        if not self.recordings:
            print("No data to save.")
            return None
        
        # Convert recordings to DataFrame
        df = pd.DataFrame(self.recordings)
        
        # Save to CSV
        df.to_csv(self.filename, index=False)
        print(f"Data saved to {self.filename}")
        return self.filename
    
    def clear(self):
        """
        Clear the current recordings and reset last_action.
        """
        self.recordings = []
        self.last_action = None
    
    def start_new_session(self):
        """
        Start a new recording session with a new session ID.
        """
        self.save()  # Save current session if there's data
        self.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.data_dir, f"eeg_gameplay_data_{self.session_id}.csv")
        
    def get_last_action(self):
        """
        Get the last recorded action.
        
        Returns:
        --------
        str or None
            The last recorded action or None if no action has been recorded yet.
        """
        return self.last_action


if __name__ == "__main__":
    # Store a reference to our connection that other modules can find
    mindwave_connection_instance = None
    
    # Example usage
    print("Initializing DataRecorder...")
    recorder = DataRecorder()
    
    if recorder.connected:
        # Share the connection instance for other modules to find
        mindwave_connection_instance = recorder.connection
        print("Running standalone recording session...")
        recorder.run_standalone()
    else:
        print("Failed to connect to Mindwave device. Check your connection.")
        if recorder.pygame_initialized:
            pygame.quit()
