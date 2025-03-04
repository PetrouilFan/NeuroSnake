import pygame
import random
import sys
import time
import argparse

# Game constants
FRAME_RATE = 4
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Controls - configurable keys
KEY_UP = pygame.K_w
KEY_DOWN = pygame.K_s
KEY_LEFT = pygame.K_a
KEY_RIGHT = pygame.K_d

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

class SnakeGame:
    def __init__(self, record_mode=False):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('NeuroSnake')
        
        # Store key configurations for external access
        self.KEY_UP = KEY_UP
        self.KEY_DOWN = KEY_DOWN
        self.KEY_LEFT = KEY_LEFT
        self.KEY_RIGHT = KEY_RIGHT
        self.FRAME_RATE = FRAME_RATE
        
        self.record_mode = record_mode
        self.reset_game()
        
    def reset_game(self):
        # Snake starts at the middle of the screen
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = "RIGHT"
        self.score = 0
        self.game_over = False
        self.food = self.spawn_food()
        
    def spawn_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food
    
    def next_frame(self, up, down, left, right):
        """
        Process the next frame of the game based on directional inputs.
        Only one direction can be True, and at least one must be True.
        """
        # Validate input - only one direction must be active
        active_dirs = sum([up, down, left, right])
        if active_dirs != 1:
            # If invalid input, maintain current direction
            if self.direction == "UP": up, down, left, right = True, False, False, False
            elif self.direction == "DOWN": up, down, left, right = False, True, False, False
            elif self.direction == "LEFT": up, down, left, right = False, False, True, False
            elif self.direction == "RIGHT": up, down, left, right = False, False, False, True
        
        # Update direction based on input (prevent 180-degree turns)
        if up and self.direction != "DOWN":
            self.direction = "UP"
        elif down and self.direction != "UP":
            self.direction = "DOWN"
        elif left and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif right and self.direction != "LEFT":
            self.direction = "RIGHT"
        
        # Move the snake
        x, y = self.snake[0]
        if self.direction == "UP":
            y -= 1
        elif self.direction == "DOWN":
            y += 1
        elif self.direction == "LEFT":
            x -= 1
        elif self.direction == "RIGHT":
            x += 1
        
        # Add new head position
        new_head = (x, y)
        
        # Check for collisions with walls
        if (x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT):
            self.game_over = True
            return self.game_over
            
        # Check for collisions with own body
        if new_head in self.snake[1:]:
            self.game_over = True
            return self.game_over
        
        # Add the new head
        self.snake.insert(0, new_head)
        
        # Check if snake eats food
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            # Remove tail if no food was eaten
            self.snake.pop()
            
        return self.game_over
        
    def draw(self):
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
        # Draw food
        pygame.draw.rect(self.screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Update display
        pygame.display.update()
    
    def run(self):
        # Main game loop for manual play
        while True:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
            # Get the key press using configurable controls
            keys = pygame.key.get_pressed()
            up = keys[KEY_UP]
            down = keys[KEY_DOWN]
            left = keys[KEY_LEFT]
            right = keys[KEY_RIGHT]
            
            # Process frame
            game_over = self.next_frame(up, down, left, right)
            
            # Check if game over
            if game_over:
                font = pygame.font.SysFont(None, 72)
                game_over_text = font.render("Game Over", True, WHITE)
                self.screen.blit(game_over_text, (WINDOW_WIDTH//2 - game_over_text.get_width()//2, 
                                                WINDOW_HEIGHT//2 - game_over_text.get_height()//2))
                pygame.display.update()
                pygame.time.wait(2000)
                self.reset_game()
            
            # Draw the game
            self.draw()
            
            # Control game speed
            self.clock.tick(FRAME_RATE)
    
    def get_state(self):
        """
        Returns the current state of the game for AI training
        """
        return {
            'snake': self.snake.copy(),
            'food': self.food,
            'direction': self.direction,
            'score': self.score
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Run NeuroSnake game')
    parser.add_argument('--record', action='store_true', help='Enable recording mode for EEG data collection')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    game = SnakeGame(record_mode=args.record)
    
    if args.record:
        # Import additional modules for recording mode
        try:
            from data_recorder import DataRecorder
            recorder = DataRecorder(game)
            recorder.run_recording_session()
        except ImportError:
            print("Error: Could not import recording modules. Running in standard mode.")
            game.run()
    else:
        game.run()