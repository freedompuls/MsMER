#!/usr/bin/env python
import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="MsMER",
    version="0.1",
    description="Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer",
    author="Yang zhao zhao",
    author_email="yangzz_work@163.com",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/freedompuls/MsMER",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    packages=find_packages(),
)
