from setuptools import setup

setup(
    name='automatic_matching',
    version='3.0.0',
    description='An automatic matching algorithm that allows to link references to the same person in multiple independent databases, based on a list of search criteria.',
    url='https://github.com/TracingThePast/automatic_matching',
    author='Jan Bernrader',
    author_email='j.bernrader@tracingthepast.org',
    license='MIT',
    packages=['automatic_matching'],
    install_requires=['numpy~=1.26',
                      'python-Levenshtein~=0.25',
                      'pyxdameraulevenshtein',
                      'rapidfuzz',
                      'doublemetaphone',
                      'PyICU~=2.13',
                      'python-dateutil~=2.8',
                      ],

    classifiers=[
        'Development Status :: 2 - Testing',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.11',
    ],
)
