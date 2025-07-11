import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="cabana",
    version="0.0.2",
    author="Gavin Lin",
    author_email="x.lin@garvan.org.au",
    description="Collagen fibre analyser for quantifying collagen fibre architecture in IHC and fluorescence microscopy images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lxfhfut/cabana.git",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    entry_points={
            'console_scripts': [
                'cabana-gui=cabana.cabana_gui:main',
            ],
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)