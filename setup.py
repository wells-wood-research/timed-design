from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="timed-predict",
    version="0.1.0",
    author="Leonardo V. Castorina, Wells Wood Research Group",
    author_email="chris.wood@ed.ac.uk",
    description="A library for using protein sequence design models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wells-wood-research/timed-predict",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    packages=find_packages(include=["utils"]),
)
