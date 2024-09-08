#!/usr/bin/env python

import os
from pathlib import Path
from setuptools import find_packages, setup


setup(
    name="expert",
    version="v0.1.0",
    description="'expert' knowledge assistant",
    author="Liam Tengelis",
    author_email="liam.tengelis@blacktuskdata.com",
    packages=find_packages(),
    package_data={
        "": ["*.yaml", "requirements.txt", "*.sql"],
        "expert": ["py.typed"],
    },
    install_requires=[
        "expert_kb",
        "expert_doc",
        "expert_llm",
        "tqdm",
    ],
    scripts=["./bin/expert"],
)
