from setuptools import setup, find_packages

setup(
    name='fairml',
    version='0.1.0',
    author='Nikhil Singh',
    author_email='nik21hil@gmail.com',
    description='Bias Detection and Mitigation Toolkit for Responsible AI',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nik21hil/fairml',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)
