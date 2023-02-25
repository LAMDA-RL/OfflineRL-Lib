

import pathlib
from setuptools import setup, find_packages

ROOT_DIR = pathlib.Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()
VERSION = (ROOT_DIR / "VERSION").read_text()

def get_install_requires():
    return [
        "gym>=0.23.1,<=0.24.1", 
        "tqdm", 
        "numpy", 
        "torch", 
        "pandas", 
        "UtilsRL"
    ]

def get_extras_require():
    return {}

setup(
    name                = "offlinerllib", 
    version             = VERSION, 
    description         = "A python module desgined for Offline RL algorithms developing and benchmarking. ", 
    long_description    = README,
    long_description_content_type = "text/markdown",
    url                 = "https://github.com/typoverflow/OfflineRLLib",
    author              = "typoverflow", 
    author_email        = "typoverflow@outlook.com", 
    license             = "MIT", 
    packages            = find_packages(),
    include_package_data = True, 
    tests_require=["pytest", "mock"], 
    python_requires=">=3.7", 
    install_requires = [
        "gym>=0.23.1,<=0.24.1", 
        "tqdm", 
        "numpy", 
        "torch", 
        "pandas", 
        "UtilsRL"
    ]
    # install_requires = get_install_requires(), 
    # extras_require = get_extras_require(), 
)