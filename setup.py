from setuptools import setup, find_packages

setup(
    name="ponyai",
    version="0.1.0",
    description="可复现、可回测、可灰度上线的量化平台",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
)
