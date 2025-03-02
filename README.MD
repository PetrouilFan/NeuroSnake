# NeuroSnake

NeuroSnake is a Python-based project that integrates an EEG headset with the classic Snake game. It allows users to generate a dataset by playing the game manually, which is then used to train an AI model capable of playing Snake using brain activity.

## Features

- **Real-time EEG Data Collection**: Connects to the MindWave Mobile headset to capture brain activity.
- **Snake Game with EEG Control**: Play Snake manually to create a dataset, then use an AI trained on EEG data to play autonomously.
- **Dataset Recording**: Saves EEG readings along with corresponding game actions.
- **AI Model Training**: Uses collected data to train an AI to predict movements based on brain activity.
- **Live Gameplay with AI Control**: The trained AI plays Snake using real-time EEG inputs.

## Components

- `mindwave.py` - Handles communication with the MindWave headset.
- `eeg_buffer.py` - Stores EEG data in a buffer for preprocessing.
- `snake_game.py` - Classic Snake game with manual and AI-controlled modes.
- `data_recorder.py` - Records EEG signals and game actions.
- `train_ai.py` - Trains a model using recorded EEG-gameplay data.
- `ai_snake.py` - Runs the trained model to play Snake using EEG inputs.

## Usage

### 1. Play Snake and Record Data

```bash
python snake_game.py --record
```

This mode allows manual play while logging EEG data and movement actions.

### 2. Train the AI Model

```bash
python train_ai.py
```

This processes the recorded dataset and trains an AI to associate EEG patterns with Snake movements.

### 3. Play Snake Using EEG Control

```bash
python ai_snake.py
```

This mode runs the trained model, allowing the AI to control the game based on live EEG inputs.

## Hardware Requirements

- NeuroSky MindWave Mobile headset
- Bluetooth adapter or built-in Bluetooth on your computer

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

