import os

# All constants

ALPHA = 0.5
BETA = 2.0
GAMMA = 0.3
M_PENALTY = 10.0
AREA_TOLERANCE = 0.01
INVERSE_ORDER_OF_TESTS = False
# PNG/videos are very expensive and do not affect the JSON solution. Enable only for debugging.
DRAW_VALIDATION_VIDEOS = True
DRAW_VALIDATION_IMAGES = True
DRAW_GROUND_TRUTH_IMAGES = False
VERBOSE_REFINEMENT = True
CALC_CONTEST_SCORE = True

FEASIBLE_PATIENCE_CHECKS = 800
N_VARIANTS = 1 # Number of starting points per checkpoint
MULTISTART_NOISE = 0.0003

# CPU<->GPU synchronization interval inside the refine loop (early-exit / progress bar).
# The best solution is now tracked on the GPU EVERY step without synchronization,
# so a large interval does not degrade quality — it only speeds up the process.
CHECK_LOSS_STEPS = 1
EARLY_EXIT_ON_ALL_FEASIBLE = True  # Set to True to exit as soon as all N variants are valid
FREEZE_ON_FEASIBLE = True  # If True, a variant is frozen as soon as a valid solution is found
MAX_POSTPROCESS_VARIANTS = 2   # how many NON-feasible variants to legalize at most (all feasible ones are processed — they are cheap). 2-3 is reasonable.
TORCH_THREADS = 1
FORCE_CPU = True
STEPS = 800
MAX_LR = 0.003
CYCLIC_LR_MULTIPLIER = 0.85
CYCLIC_LR_PERIODS = 10

R_SCALE = 1.2
USE_AMP = False

PROCESS_TESTS = None
# PROCESS_TESTS = [98, 100]

# Train part
# Losses block
USE_BATCHED_MSE = True
USE_PER_ITEM_LOSSES = True
USE_PAIRWISE_LOSS = True
USE_PHYS_LOSS = True
BATCH_SIZE = 128
TORCH_NUM_THREADS = 1
LAMBDA_PHYS_LOSS = 2.0


# DATA_ROOT = 'D:/Projects/2026_05_ICCAD2026_ProblemC/'
# DATA_ROOT = './'
DATA_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'
