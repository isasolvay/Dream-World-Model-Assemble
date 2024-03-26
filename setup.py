import setuptools
import pathlib

extras = {}

setuptools.setup(
    name='pydreamer',
    version='0.1.0',
    packages=['pydreamer', 'pydreamer.models', 'pydreamer.envs'],
    author='Jurgis Pašukonis',
    author_email='jurgisp@gmail.com',
    description='DreamerV2 in PyTorch',
    url='http://github.com/jurgisp/pydreamer',
    long_description=pathlib.Path('README.md').read_tex