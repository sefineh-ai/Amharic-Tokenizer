"""Setup script for the Amharic Tokenizer package with Cython acceleration."""
import os
from Cython.Build import cythonize
from setuptools import Extension, setup # type: ignore

here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

extensions = [
    Extension(
        name="amharic_tokenizer.tokenizer",
        sources=["amharic_tokenizer/tokenizer.pyx"],
        language="c++",
    )
]

setup(
    name="amharic-tokenizer",
    version="0.2.1",
    author="Sefineh Tesfa",
    author_email="sefinehtesfa34@gmail.com",
    description=(
        "A robust Amharic tokenizer with vowel decomposition and Cython "
        "acceleration"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["amharic_tokenizer"],
    ext_modules=cythonize(extensions, language_level="3"),
    # Console scripts are provided via pyproject.toml [project.scripts]
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
)
