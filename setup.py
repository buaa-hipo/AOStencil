from setuptools import setup, find_packages

setup(
    name="AOStencil",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "dev": [],
    },
    author="somehow6",
    author_email="somehow6@buaa.edu.cn",
    description="High-performance stencil computation dsl compiler for ARM-based processors",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/buaa-hipo/AOStencil",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: POSIX :: Linux",
    ],
)
