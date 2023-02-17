from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='fly-pipe',
    description='Fruit fly trajectory analysis',
    version=0.0,
    long_description=long_description,
    url='',
    author='',
    author_email='',
    maintainer='',
    maintainer_email='',
    packages=[
        'fly_pipe',
        'fly_pipe.utils'
    ],
    classifiers=[
        'Topic :: Drosophila Melanogaster',
    ],
    keywords=['trajectory', 'data-analysis'],
    package_dir={'fly_pipe': 'fly_pipe'},
    install_requires=[
        'pandas>1.5'
    ],
    project_urls=dict(
        Documentation='',
        Source='',
        Issues='',
        Changelog=''
    )

)
