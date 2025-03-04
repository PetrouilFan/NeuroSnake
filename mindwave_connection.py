# Add the EEGBuffer class at the top
class EEGBuffer:
    def __init__(self):
        self.buffer = []
        self.current_window = []
    def on_value_callback(self, **kwargs):
        # Append each value update as a dictionary
        self.buffer.append(kwargs)
        self.current_window.append(kwargs)

import time
import threading
import mindwave

DEFAULT_COM_PORT = "COM4"

class MindwaveConnection:
    """
    Reliable connection manager for MindWave EEG headset.
    Handles data buffering and provides easy access to different data types.
    """
    def __init__(self, port=DEFAULT_COM_PORT, window_size=1.0):
        """
        Initialize the connection manager.
        
        Parameters:
        -----------
        port : str
            Serial port name (e.g., COM4 on Windows, /dev/ttyUSB0 on Linux)
        window_size : float
            Size of the data window in seconds
        """
        self.port = port
        self.headset = None
        self.connected = False
        self.running = False
        self.window_size = window_size
        
        # Buffer for raw EEG data, attention, meditation, etc.
        self.raw_buffer = []
        self.attention_buffer = []
        self.meditation_buffer = []
        self.blink_buffer = []
        self.signal_quality_buffer = []
        self.waves_buffer = {}  # Dict for wave types
        
        # Time tracking
        self.start_time = 0
        self.last_update = 0
        
        # Current values
        self.raw_value = 0
        self.attention = 0
        self.meditation = 0
        self.blink = 0
        self.poor_signal = 0
        self.waves = {}
        
        # Data storage using windows approach
        self.window_buffer = []
        self.current_window = []
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Processing state
        self.collect_thread = None
    
    def connect(self):
        """
        Connect to the MindWave headset.
        
        Returns:
        --------
        bool
            True if connection was successful, False otherwise
        """
        try:
            print(f"Attempting to connect to Mindwave on port {self.port}...")
            
            from mindwave import Headset
            self.headset = Headset(self.port)
            
            # Register callbacks to receive data
            if hasattr(self.headset, 'attention_handlers'):
                self.headset.attention_handlers.append(self._on_attention)
            if hasattr(self.headset, 'meditation_handlers'):
                self.headset.meditation_handlers.append(self._on_meditation)
            if hasattr(self.headset, 'blink_handlers'):
                self.headset.blink_handlers.append(self._on_blink)
            if hasattr(self.headset, 'raw_value_handlers'):
                self.headset.raw_value_handlers.append(self._on_raw_value)
            if hasattr(self.headset, 'poor_signal_handlers'):
                self.headset.poor_signal_handlers.append(self._on_poor_signal)
            if hasattr(self.headset, 'wave_handlers'):
                self.headset.wave_handlers.append(self._on_waves)
            
            print(f"Successfully connected to Mindwave device on {self.port}")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Error connecting to Mindwave device: {e}")
            self.headset = None
            self.connected = False
            return False
    
    def start(self):
        """Start collecting data from the headset"""
        if not self.connected or not self.headset:
            if not self.connect():
                print("Cannot start: Failed to connect to headset")
                return False
        
        self.running = True
        self.start_time = time.time()
        self.last_update = self.start_time
        
        # Initialize buffers
        self.window_buffer = []
        self.current_window = []
        
        # Start the data collection thread
        self.collect_thread = threading.Thread(target=self._collect_data)
        self.collect_thread.daemon = True
        self.collect_thread.start()
        
        return True
    
    def stop(self):
        """Stop collecting data and disconnect"""
        self.running = False
        # Wait for the collection thread to finish if it's running
        if hasattr(self, 'collect_thread') and self.collect_thread.is_alive():
            self.collect_thread.join(timeout=1.0)
        self.disconnect()
    
    def disconnect(self):
        """Safely disconnect from the headset"""
        if self.headset:
            try:
                if hasattr(self.headset, 'disconnect'):
                    self.headset.disconnect()
                
                # Stop the listener thread
                if hasattr(self.headset, 'listener') and self.headset.listener:
                    self.headset.listener.running = False
                    # Give the thread time to exit
                    time.sleep(0.2)
                
                # Close the serial connection
                if hasattr(self.headset, 'dongle') and self.headset.dongle:
                    self.headset.dongle.close()
                
                print("Disconnected from Mindwave headset")
            except Exception as e:
                print(f"Error during disconnect: {e}")
        
        self.connected = False
    
    def _collect_data(self):
        """Background thread to collect data in windows"""
        window_start_time = time.time()
        current_samples = []
        
        while self.running:
            # Calculate elapsed time since window start
            now = time.time()
            elapsed = now - window_start_time
            
            # If window is complete, add it to buffer and start new window
            if elapsed >= self.window_size:
                with self.lock:
                    self.window_buffer.append(current_samples.copy())
                    # Keep only the last 10 windows
                    if len(self.window_buffer) > 10:
                        self.window_buffer.pop(0)
                    # Reset current window
                    current_samples = []
                
                # Start a new window
                window_start_time = now
            
            # Add current values to current window
            with self.lock:
                sample = (self.raw_value, self.attention, self.meditation, self.blink, self.poor_signal, self.waves.copy())
                current_samples.append(sample)
                self.current_window = current_samples.copy()
            
            # Sleep briefly to avoid high CPU usage
            time.sleep(0.02)
    
    def get_buffer(self):
        """
        Get a copy of the complete window buffer.
        
        Returns:
        --------
        list
            List of data windows, each window containing samples
        """
        with self.lock:
            return self.window_buffer.copy()
    
    def get_current_window(self):
        """
        Get the current (incomplete) data window.
        
        Returns:
        --------
        list
            Current data window with samples
        """
        with self.lock:
            return self.current_window.copy()
    
    def get_signal_quality(self):
        """
        Get the current signal quality.
        
        Returns:
        --------
        int
            Current signal quality value (0-200, lower is better)
        """
        return self.poor_signal
    
    def _on_attention(self, headset, value):
        """Handler for attention value updates"""
        self.attention = value
        
    def _on_meditation(self, headset, value):
        """Handler for meditation value updates"""
        self.meditation = value
    
    def _on_blink(self, headset, value):
        """Handler for blink value updates"""
        self.blink = value
    
    def _on_raw_value(self, headset, value):
        """Handler for raw EEG value updates"""
        self.raw_value = value
    
    def _on_poor_signal(self, headset, value):
        """Handler for poor signal updates"""
        self.poor_signal = value
    
    def _on_waves(self, headset, waves):
        """Handler for brainwave data updates"""
        self.waves = waves.copy() if waves else {}