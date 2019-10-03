from setuptools import setup

setup(name='pymtrf',
      version='0.0.1a0',
      description='Translation of the matlab mtrf toolbox, see http://www.mee.tcd.ie/lalorlab/resources.html',
      url='http://github.com/storborg/funniest',
      author='Simon R. Steinkamp',
      author_email='simon.steinkamp@googlemail.com',
      license='tbd',
      packages=['pymtrf'],
	install_requires=['scipy>=0.11.0', 'numpy>=1.14.0', 'pytest==5.0.1'],
	python_requires='>3.6.0',
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False)