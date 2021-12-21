from setuptools import find_packages, setup

# with open("README.md", "r") as f:
#     long_description = f.read()

setup(
    name="NNHedge",
    version="0.1.0",
    author="Guijin Son",
    author_email="spthsrbwls123@yonsei.ac.kr",
    description="A Deep Learning Framework for Neural Derivative Hedging",
#    long_description=long_description,
#   long_description_content_type="text/markdown",
    url="https://github.com/guijinSON/NNHedge",
#    packages=find_packages(exclude=["docs", "tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "matplotlib","torch"],
)
