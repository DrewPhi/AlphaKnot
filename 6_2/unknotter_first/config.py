# config.py

# === Core Training Parameters ===
num_iterations = 40           # Number of training iterations
num_episodes = 100            # Self-play games per iteration
num_mcts_sims = 800           # Increased from 75 for deeper MCTS
num_epochs = 20               # NN epochs per training step
learning_rate = 0.001         # Adam optimizer LR

# === Save & Resume Options ===
checkpoint_path = './models/6_2/'
save_model_freq = 5
resumeTraining = False
load_model = False
load_model_file = 'best.pth.tar'

# === Game Rules & Evaluation ===
knotter_first = False
n_games = 100  # Number of head-to-head games between champion & candidate
require_perfect_random_win = True

# === Optional additions ===
dirichlet_alpha = 0.3
dirichlet_epsilon = 0.25
# === Promotion Policy Toggle ===
promote_by_loss_only = True  # Set to True to mimic AlphaZero-GitHub behavior