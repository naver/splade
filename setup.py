from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='SPLADE',
    version='2.1',
    description='SParse Lexical AnD Expansion Model for First Stage Ranking',
    url='https://github.com/naver/splade',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=['splade'],
    license="Creative Commons Attribution-NonCommercial-ShareAlike",
    long_description=readme,
    install_requires=[
        'transformers==4.18.0', 
    ],
)