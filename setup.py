from setuptools import setup, find_packages

setup(
	name='project2',
	version='1.0',
	author='Catherine Donner',
	authour_email='Catherine.G.Donner-1@ou.edu',
	packages=find_packages(exclude=('tests', 'docs')),
	setup_requires=['pytest-runner'],
	tests_require=['pytest']	
)
