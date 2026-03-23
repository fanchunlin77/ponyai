from setuptools import setup, find_packages

setup(
    name="ponyai",
    version="0.1.0",
    description="A reproducible, backtestable, and canary-deployable quantitative trading platform",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
        ],
    },
)
