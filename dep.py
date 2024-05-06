import torch
from agent import Agent
from game import SnakeGameAI

# Load the trained model
model = Agent()
model.model.load_state_dict(torch.load("C:\\Users\\Lenovo\\Downloads\\PythonRlgame\\model\\model_20240506_171650.pth"))
model.model.eval()  # Set the model to evaluation mode

# Initialize the game
game = SnakeGameAI()

while True:
    # Get the current game state
    state = model.get_state(game)
    
    # Get the action from the model
    action = model.get_action(state)
    
    # Perform the action and get the reward, game over status, and score
    reward, game_over, score = game.play_step(action)
    
    # If the game is over, reset the game
    if game_over:
        game.reset()
        continue
