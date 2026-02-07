import setuptools
from setuptools import setup

NAME = "tf_exporter"
print('packages found', setuptools.find_packages())


def load_dependencies():
    """Loads a limited set of dependencies, with
    serving in mind."""
    with open("requirements.txt") as f:
        required = f.read().splitlines()
    return required  # Should be List[str] for install_requires


setup(
    name=NAME,
    packages=setuptools.find_packages(),
    version="0.0.1",
    description='Packages a sentence-transformer models as a single tensorflow graph',
    python_requires='>=3.10',
    install_requires=load_dependencies(),
)