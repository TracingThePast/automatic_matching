from setuptools import setup

setup(
    name='automatic_matching',
    version='0.1.4',    
    description='An automatic matching algorithm that allows to link references to the same person in multiple independent databases, based on a list of search criteria.',
    url='https://github.com/TracingThePast/automatic_matching',
    author='Jan Bernrader',
    author_email='j.bernrader@tracingthepast.org',
    license='MIT',
    packages=['automatic_matching'],
    install_requires=['levenshtein~=0.25',
                      'numpy~=1.26',    
                      'pyicu~=2.13.1'                 
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT',  
        'Operating System :: POSIX :: Linux',   
        'Programming Language :: Python :: 3.11',
    ],
)