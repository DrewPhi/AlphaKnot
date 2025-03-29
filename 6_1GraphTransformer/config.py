pd_codes = [
    [[1,7,2,6],[3,10,4,11],[5,3,6,2],[7,1,8,12],[9,4,10,5],[11,9,12,8]]
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
knotter_first = False