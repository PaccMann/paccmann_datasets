"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ''
if os.path.exists('README.md'):
    with open('README.md') as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name='pytoda',
    version='0.0.3',
    description='pytoda: PaccMann PyTorch Dataset Classes.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Matteo Manica, Jannis Born, Ali Oskooei, Joris Cadow',
    author_email=(
        'drugilsberg@gmail.com, jab@zurich.ibm.com, '
        'ali.oskooei@gmail.com, joriscadow@gmail.com'
    ),
    url='https://github.com/PaccMann/paccmann_datasets',
    license='MIT',
    install_requires=[
        'numpy', 'scikit-learn', 'pandas', 'torch>=1.0.0', 'diskcache', 'dill',
        'selfies'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages(),
    scripts=scripts
)
