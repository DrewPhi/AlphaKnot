# Training parameters
num_iterations = 1          # Total number of training iterations
num_episodes = 100          # Number of self-play games per iteration
num_mcts_sims = 25          # Number of MCTS simulations per move
learning_rate = 0.001       # Learning rate for the neural network
num_epochs = 10             # 🔥 NEW: Number of epochs per training step
checkpoint_path = './models/'
load_model = False
load_model_file = 'best.pth.tar'
save_model_freq = 100
# Existing parameters
knotter_first = False
n_games = 100

# New parameter
resumeTraining = False
