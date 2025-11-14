"""
Setup script for Amharic Tokenizer (pure Python version without Cython).
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="amharic-tokenizer",
    version="0.3.0",
    author="Sefineh Tesfa",
    author_email="sefinehtesfa34@gmail.com",
    description="A robust Amharic tokenizer with vowel decomposition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Text Processing :: Linguistic",
    ],
)
