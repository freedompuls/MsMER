#!/usr/bin/env python
import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="MsMER",
    description="MsMER: A Multi-Scale Feature for Transformer-based Handwritten Mathematical Expression Recognition",
    author="Yang Zhao Zhao",
    author_email="490811656@qq.com",

    url="https://github.com/freedompuls/MsMER",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    packages=find_packages(),
)
