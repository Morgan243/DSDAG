from setuptools import setup

setup(name='dsdag',
      version='0.2',
      description='Data Science DAG Processing',
      author='Morgan Stuart',
      packages=['dsdag', 'dsdag/core', 'dsdag/ext'],#['dsdag', 'projects'],
      install_requires=['toposort', 'imblearn', 'sklearn',
                        'pandas', 'numpy', 'pydot', 'graphviz', 'dask',
                        'matplotlib', 'tqdm', 'psutil', 'attrs', 'ipywidgets'],
      zip_safe=False
      )