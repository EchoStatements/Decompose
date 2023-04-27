from distutils.core import setup


if __name__ == '__main__':

    setup(
        name='decompose',
        version='0.1.0',
        author='Danny Wood & Andrew M. Webb',
        author_email='danny.wood@manchester.ac.uk',
        packages=['decompose'],
        license='MIT License',
        description='A package for decomposing Bregman divergences and other losses',
        python_requires='>=3.4.3',
        install_requires=[
            "numpy >= 1.11.3",
            "scipy",
            "sklearn",
            "matplotlib",
            "tqdm",
            "cached_property"
        ],
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
    )
