


num_iterations = 1          # Total number of training iterations
num_episodes = 100             # Number of self-play games per iteration
num_mcts_sims = 25             # Number of MCTS simulations per move
learning_rate = 0.001          # Learning rate for the neural network
checkpoint_path = './models/'  # Directory to save model checkpoints
load_model = False             # Whether to load an existing model (used for evaluation)
load_model_file = 'best.pth.tar'  # File name of the model to load
save_model_freq = 100          # Save the model every N iterations


knotter_first = False          # If True, the knotter (Player -1) goes first
n_games = 100                  # Number of games to play for evaluation scripts

resumeTraining = True         # If True, resume training from the current model
