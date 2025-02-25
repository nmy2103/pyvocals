from setuptools import setup, find_packages

setup(
    name = 'pyvocals',
    version = '0.1',
    author = 'Natasha Yamane',
    author_email = 'natasha.yamane@gmail.com',
    packages = find_packages(),
    install_requires = [
        'numpy==1.26.4',
        'pandas==2.2.2',
        'matplotlib>=3.9',
        'librosa>=0.10.2',
    ],
    python_requires = '>=3.11',
)