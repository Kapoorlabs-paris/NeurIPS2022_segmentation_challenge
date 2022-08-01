import setuptools
from setuptools import  setup
import os


_dir = os.path.dirname(__file__)

with open('README.md') as f:
    long_description = f.read()
with open(os.path.join(_dir,'neurips','_version.py'), encoding="utf-8") as f:
    
    exec(f.read())
    print(exec(f.read()))
setup(
    name="neurips",

    version=__version__,

    author='Varun Kapoor',
    author_email='varun.kapoor@kapoorlabs.org',
    url='https://github.com/Kapoorlabs-paris/NeurIPS2022_segmentation_challenge/',
    description='Kapoorlabs submission to the segmentation challenge.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        
        "vollseg",
        "hydra-core"
        
       
    ],
    
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
    ],
)
