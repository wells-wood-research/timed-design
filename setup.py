from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="timed-design",
    version="1.0.0",
    author="Wells Wood Research Group",
    author_email="chris.wood@ed.ac.uk",
    description="Protein Sequence Design with Deep Learning and Tooling like Monte Carlo Sampling and Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    url="https://github.com/wells-wood-research/timed-design",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["design_utils"],
    install_requires=[
        "altair==4.2.0",
        "ampal==1.5.1",
        "aposteriori==2.3.0",
        "logomaker==0.8",
        "matplotlib==3.5.1",
        "millify==0.1.1",
        "py3Dmol==2.0.0.post2",
        "requests==2.31.0",
        "scikit_learn==1.1.2",
        "seaborn==0.11.2",
        "setuptools>=65.5.1",
        "stmol==0.0.9",
        "streamlit==1.30.0",
        "tensorflow==2.13.0",
        "tqdm==4.64.0",
    ],
)
