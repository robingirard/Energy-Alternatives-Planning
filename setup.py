from setuptools import setup
import os
#TODO utiliser Pathlib

setup(
    name='EnergyAlternativesPlanning',
    version='0.1.3',
    description='A Python package for energy system modeling',
    url='https://github.com/robingirard/Energy-Alternatives-Planning',
    author='Robin Girard',
    author_email='robin.girard@minesparis.psl.eu',
    license='MIT License',
    packages=['EnergyAlternativesPlanning'],
    install_requires=['pandas',
                      'numpy',
                      'pyomo',
                      'Mosek',
                      'plotly',
                      'sklearn',
                      'mycolorpy'],
    package_data={"EnergyAlternativesPlanning" : ["*.csv", "*.xlsx"]},
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)