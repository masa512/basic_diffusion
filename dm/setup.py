from setuptools import setup, find_packages

setup(
    name="dm",
    version="0.1.1",
    packages=find_packages(),  # Finds "project" automatically
    install_requires=[
        "numpy",  # Add dependencies here
        "scipy",
        "torch"]
)