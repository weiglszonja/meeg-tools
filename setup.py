# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

# Get requirements
with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup_args = dict(
    name='meeg-tools',
    version='0.3.2',
    description='EEG/MEEG data preprocessing and analyses tools',
    python_requires='<3.10.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Szonja Weigl',
    author_email='weigl.anna.szonja@gmail.com',
    url='https://github.com/weiglszonja/meeg-tools',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
)


if __name__ == '__main__':
    setup(**setup_args)
