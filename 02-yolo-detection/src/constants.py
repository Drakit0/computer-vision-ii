"""
Script to declare the constants used in the project.
"""

EPSILON = 1e-6

TARGET_SIZE_IMG = (400, 400)

S = 7
B = 2
C = 1

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.25

# Architecture of the CNN. Each tuple represents (kernel, out_channels, stride, padding)
# The strings ("M") represent 2 x 2 Max Pooling.
MODEL: list[tuple[int, int, int, int] | str] = [
    (7, 8, 2, 3),
    "M",
    (3, 16, 1, 1),
    "M",
    (1, 16, 1, 0),
    (3, 32, 1, 1),
    (3, 64, 1, 1),
    "M",
    (1, 32, 1, 0),
    (3, 64, 1, 1),
    "M",
    (3, 64, 1, 0),
    (3, 128, 1, 1),
]

LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 45
PATH_MODEL = "weights/YOLO.pt"
