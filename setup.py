"""Package installer."""
import codecs
import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


LONG_DESCRIPTION = ''
if os.path.exists('README.md'):
    with open('README.md') as fp:
        LONG_DESCRIPTION = fp.read()

scripts = ['bin/pytoda-filter-invalid-smi']

setup(
    name='pytoda',
    version=get_version('pytoda/__init__.py'),
    description='pytoda: PaccMann PyTorch Dataset Classes.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Matteo Manica, Jannis Born, Joris Cadow',
    author_email='drugilsberg@gmail.com, jannis.born@gmx.de, joriscadow@gmail.com',
    url='https://github.com/PaccMann/paccmann_datasets',
    license='MIT',
    install_requires=[
        'numpy',
        'scikit-learn',
        'pandas',
        'torch>=1.4.0',
        'diskcache',
        'dill>=0.3.3',
        'selfies>=2.0.0',
        'upfp',
        'SmilesPE>=0.0.3',
        'pyfaidx',
        'pubchempy',
        'importlib_resources',
        'Unidecode',
        'rdkit-pypi>=2021.9.3',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(),
    package_data={
        'pytoda.smiles.metadata': [
            'spe_chembl.txt',
            'ATTRIBUTION',
            'README.md',
            'tokenizer/*',
        ],
        'pytoda.proteins.metadata': ['kinase_activesite_alignment.smi'],
    },
    scripts=scripts,
)
