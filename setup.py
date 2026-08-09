from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='youtube_viewer_sentiment_analysis',
    version='0.0.1',
    author='A.Gireesh kumar Gowd',
    author_email='gireeshkumargowd@gmail.com',
    packages=find_packages(include=["src*"]),
    install_requires=get_requirements('requirements.txt')
)