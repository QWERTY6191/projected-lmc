import setuptools

# Load the long_description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="projectedlmc",
    version="0.0.1",
    author="",
    author_email="",
    description="A short package based on gpytorch implementing the Projected LMC model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://qwerty6191.github.io/projected-lmc/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

