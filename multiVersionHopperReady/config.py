# config.py

# Training parameters
num_iterations = 1          # Total number of training iterations
num_episodes = 100             # Number of self-play games per iteration
num_mcts_sims = 25             # Number of MCTS simulations per move
learning_rate = 0.001          # Learning rate for the neural network
checkpoint_path = './models/'  # Directory to save model checkpoints
load_model = False             # Whether to load an existing model (used for evaluation)
load_model_file = 'best.pth.tar'  # File name of the model to load
save_model_freq = 100          # Save the model every N iterations
# Parallel MCTS Settings
num_parallel_mcts = 8  # Adjust based on CPU cores
c_puct = 1.0  # Exploration constant
# Parallel execution settings
num_parallel_games = 8  # Adjust based on the number of CPU cores available

# Existing parameters
knotter_first = False          # If True, the knotter (Player -1) goes first
n_games = 100                  # Number of games to play for evaluation scripts

# New parameter
resumeTraining = True         # If True, resume training from the current model
