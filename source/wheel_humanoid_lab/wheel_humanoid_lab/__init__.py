import os

WHEEL_HUMANOID_LAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WHEEL_HUMANOID_ROOT_DIR = os.path.abspath(
    os.environ.get(
        "WHEEL_HUMANOID_ROOT",
        os.path.join(WHEEL_HUMANOID_LAB_DIR, "..", ".."),
    )
)
