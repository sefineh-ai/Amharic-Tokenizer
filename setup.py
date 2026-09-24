"""Build configuration for the Cython extension; all metadata lives in pyproject.toml."""

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="amharic_tokenizer._bpe",
        sources=["src/amharic_tokenizer/_bpe.pyx"],
        language="c++",
    )
]

setup(ext_modules=cythonize(extensions, language_level="3"))
