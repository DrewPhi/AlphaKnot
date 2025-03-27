# config.py

# === Core Training Parameters ===
num_iterations = 40           # Number of training iterations
num_episodes = 100            # Self-play games per iteration
num_mcts_sims = 75            # MCTS rollouts per move
num_epochs = 20               # NN epochs per training step
learning_rate = 0.001         # Adam optimizer LR

# === Save & Resume Options ===
checkpoint_path = './models/6_2/'
save_model_freq = 5
resumeTraining = True
load_model = False
load_model_file = 'best.pth.tar'

# === Game Rules & Evaluation ===
knotter_first = True
n_games = 100  # Number of head-to-head games between champion & candidate
require_perfect_random_win = True  
# If True, training extends beyond num_iterations until champion wins 200/200 vs random.
# If champion gets 200/200 early, training still does the planned num_iterations.

# === Optional additions ===
# batch_size = 64
# dirichlet_alpha = 0.3
# dirichlet_epsilon = 0.25
