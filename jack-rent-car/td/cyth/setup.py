from setuptools import setup
from Cython.Build import cythonize

setup(
    name='jackrentcar',
    ext_modules=cythonize(["jackrentcar.pyx"]),
    zip_safe=False,
)
