import os


pd_codes = [
#    	[[1,5,2,4],[3,1,4,6],[5,3,6,2]],
#[[4,2,5,1],[8,6,1,5],[6,3,7,4],[2,7,3,8]],
#[[2,8,3,7],[4,10,5,9],[6,2,7,1],[8,4,9,3],[10,6,1,5]],
#[[1,5,2,4],[3,9,4,8],[5,1,6,10],[7,3,8,2],[9,7,10,6]],
#[[1,7,2,6],[3,10,4,11],[5,3,6,2],[7,1,8,12],[9,4,10,5],[11,9,12,8]],
#[[1,8,2,9],[3,11,4,10],[5,1,6,12],[7,2,8,3],[9,7,10,6],[11,5,12,4]],
#[[4,2,5,1],[8,4,9,3],[12,9,1,10],[10,5,11,6],[6,11,7,12],[2,8,3,7]],
[[1,9,2,8],[3,11,4,10],[5,13,6,12],[7,1,8,14],[9,3,10,2],[11,5,12,4],[13,7,14,6]],
#[[2,10,3,9],[4,14,5,13],[6,12,7,11],[8,2,9,1],[10,8,11,7],[12,6,13,5],[14,4,1,3]],
#[[1,9,2,8],[3,11,4,10],[5,1,6,14],[7,13,8,12],[9,3,10,2],[11,5,12,4],[13,7,14,6]],
#[[2,10,3,9],[4,12,5,11],[6,14,7,13],[8,4,9,3],[10,2,11,1],[12,8,13,7],[14,6,1,5]],
#[[2,10,3,9],[4,2,5,1],[6,14,7,13],[8,12,9,11],[10,4,11,3],[12,6,13,5],[14,8,1,7]],
#[[1,13,2,12],[3,9,4,8],[5,1,6,14],[7,10,8,11],[9,3,10,2],[11,6,12,7],[13,5,14,4]],
#[[1,10,2,11],[3,13,4,12],[5,14,6,1],[7,5,8,4],[9,2,10,3],[11,9,12,8],[13,6,14,7]]
]
max_strand_label = 14  # PD edge labels run from 1 through 2 * crossings.
max_validated_crossings = 7

numIters = 1000  # Start with 100, scale up if needed.
numEps = 100
tempThreshold = 10
updateThreshold = 0.55
maxlenOfQueue = 50000
numMCTSSims = 200
cpuct = 1.0
num_epochs = 10

checkpoint = './checkpoints/'
saveIterCheckpoint = True
load_model = False
train_examples_path = None
arenaCompare = 4
knotter_first = True
randomGames = 50
resume_training = os.environ.get("ALPHAKNOT_RESUME", "0") == "1"
random_play_fraction = 0  # 25% of self-play games use a random player
selfplay_device = "cpu"
arena_device = "cpu"
selfplay_workers = 0  # 0 = use SLURM_CPUS_PER_TASK
arena_workers = 0     # 0 = use SLURM_CPUS_PER_TASK


def validate():
    if not pd_codes:
        raise ValueError("At least one PD code is required")
    crossing_counts = {len(pd_code) for pd_code in pd_codes}
    if len(crossing_counts) != 1:
        raise ValueError("All production PD codes must have the same size")
    if max(crossing_counts) > max_validated_crossings:
        raise ValueError("PD code exceeds max_validated_crossings")
    actual_max_label = max(
        label for pd_code in pd_codes for crossing in pd_code for label in crossing
    )
    if actual_max_label > max_strand_label:
        raise ValueError("max_strand_label is too small for configured PD codes")
    if arenaCompare > 0 and (arenaCompare < 2 or arenaCompare % 2):
        raise ValueError("arenaCompare must be zero or a positive even number")
