"""Isaac Lab extension for dual-wheel humanoid skateboard locomotion."""

from setuptools import find_packages, setup

setup(
    name="wheel_humanoid_lab",
    packages=find_packages(),
    version="0.1.0",
    description="Isaac Lab PPO skateboard sliding for the dual-wheel humanoid URDF",
    install_requires=["psutil"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
