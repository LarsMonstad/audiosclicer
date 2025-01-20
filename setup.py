from setuptools import setup, find_packages
setup(
    name="lm-audioslicer",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        'librosa==0.9.1',
        'soundfile>=0.12.1',
        'numpy==1.23.5',
    ],
    author="Lars Monstad",
    author_email="Lars.Monstad@gmail.com",
    description="A tool for slicing audio files based on silence detection",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LarsMonstad/audioslicer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'audioslicer=audioslicer.slicer:main',
        ],
    }
)
