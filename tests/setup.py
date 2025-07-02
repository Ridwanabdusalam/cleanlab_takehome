from setuptools import setup, find_packages

setup(
    name="trustworthiness-detector",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "litellm",
        "python-dotenv>=1.0.0",
    ],
)