"""
Setup script for virchowstr6_i4b package
"""

from setuptools import setup, find_packages

setup(
    name='virchowstr6_i4b',
    version='1.0.0',
    description='i4b-style heating system for Virchowstr. 6 with 4R3C building model',
    author='Virchowstr. 6 Project',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'openpyxl>=3.0.0',
    ],
    python_requires='>=3.8',
)
