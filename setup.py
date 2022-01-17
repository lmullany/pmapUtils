from setuptools import setup, find_packages

setup(
    name="pmap_dbapi",
    version="1.0",
    packages=find_packages(),
    url="",
    license="",
    author="JHU APL",
    author_email="",
    description="",
    install_requires=[
        "pandas",
        "scikit-learn",
        "sqlalchemy",
        "pyyaml",
        "pyodbc",
        "matplotlib",
        "plotly",
        "tqdm",
        "tabulate",
        "click",
        "docstring-parser",
        "datatable",
    ],
)
