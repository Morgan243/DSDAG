from setuptools import setup

setup(name='dsdag',
      version='0.2',
      description='Data Science DAG Processing',
      author='Morgan Stuart',
      packages=['dsdag', 'dsdag/core', 'dsdag/ext'],#['dsdag', 'projects'],
      install_requires=['toposort'],
      zip_safe=False
      )