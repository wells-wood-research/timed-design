from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="timed-design",
    version="0.1.0",
    author="Leonardo V Castorina, Wells Wood Lab, University of Edinburgh",
    author_email="leonardo.castorina@ed.ac.uk",
    description="Protein Sequence Design with Deep Learning and Tooling like Monte Carlo Sampling and Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wells-wood-research/timed-design",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['design_utils'],
)
