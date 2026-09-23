from setuptools import setup

setup(name='miletos',
      version='0.1',
      description='miletos',
    url='https://github.com/tansudaylan/miletos',
      author='Tansu Daylan',
      author_email='tansu.daylan@gmail.com',
      license='MIT',
      packages=['miletos'],
      install_requires=[
          'numpy', 'scipy', 'gputil', \
      ],
      zip_safe=False)
