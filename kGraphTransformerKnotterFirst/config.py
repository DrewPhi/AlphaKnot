pd_codes = [
    [[2,5,3,6],[4,10,5,9],[6,11,7,12],[8,1,9,2],[10,4,11,3],[12,7,1,8]]
]

numIters = 100  # Start with 100, scale up if needed.
numEps = 50
tempThreshold = 10
updateThreshold = 0.55
maxlenOfQueue = 50000
numMCTSSims = 200
cpuct = 1.0
num_epochs = 10

checkpoint = './checkpoints/'
saveIterCheckpoint = True
load_model = False
arenaCompare = 20
knotter_first = True

resume_training = True  # or False
random_play_fraction = 0  # 25% of self-play games use a random player
