from setuptools import setup, find_packages
import os

name='my_utils'
version='0.1.0'
description='自分で使う用の関数'
author='shusuke_machida'
# author_email=''
url=''


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


dependency_links = [

]


setup(
    name=name,
    version=version,
    description=description,
    author=author,
    # install_requires=read_requirements(),
    url=url,
    packages=["my_utils"],
    test_suite='tests',
    include_package_data=True,
    zip_safe=False
)
