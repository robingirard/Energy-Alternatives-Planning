from setuptools import setup

setup(
    name='EnergyAlternativesPlaning',
    version='0.1.1',
    description='A Python package for energy system modeling',
    url='https://github.com/robingirard/Energy-Alternatives-Planing',
    author='Robin Girard',
    author_email='robin.girard@minesparis.psl.eu',
    license='MIT License',
    packages=['EnergyAlternativesPlaning'],
    install_requires=['pandas',
                      'numpy',
                      'pyomo',
                      'Mosek',
                      'plotly',
                      'matplotlib',
                      'sklearn',
                      'mycolorpy'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)