#!/usr/bin/env python

from distutils.core import setup
import sys
import os

try:
    assert sys.version_info == (3, 7)
except:
    print('Python version not compatible. Please use python 3.7')

if not os.path.exists('./logs'):
    os.mkdir('./logs')

package_list = [
    'networkx',
    'numpy',
    'scipy',
    'shapely',
    'matplotlib',
    'seaborn',
]

setup(name='muar-sfc',
      version='1.0',
      description='Python Distribution Utilities',
      author='Hugo Leonardo',
      author_email='hugosantos@ufpa.br',
      url='github.com/muar',
      packages=package_list
     )