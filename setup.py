from setuptools import setup

setup(
    name='sde_simulator',
    version='0.1.6',
    description="General base class implementation of Rung-Kutta SDE solver",
    long_description="This package provides a base class for simulating SDE using Rung-Kutta method.\n"
                     "The base class implements the simulation and requires the override of the deterministic and "
                     "stochastic methods of the model.",
    author="Oded Wertheimer",
    author_email="oded.wertheimer@mail.huji.ac.il",
    packages=["sde_simulator"],
    install_requires=["numpy", "tqdm"],
    license="MIT",
    url='https://github.com/odedwer/sde_simulator',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
