# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

# Get requirements
with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup_args = dict(
    name='eeg-preprocessing',
    version='0.2.4',
    description='Semiautomatic framework for preprocessing EEG data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Szonja Weigl',
    author_email='weigl.anna.szonja@gmail.com',
    url='https://github.com/weiglszonja/eeg-preprocessing',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
)


if __name__ == '__main__':
    setup(**setup_args)
