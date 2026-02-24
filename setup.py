from setuptools import setup, find_packages

setup(
    name="gym_air_traffic",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "gymnasium",
        "pygame",
        "numpy",
        "imageio[ffmpeg]"
    ]
)