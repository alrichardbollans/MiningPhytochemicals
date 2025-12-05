from setuptools import setup, find_packages

setup(
    name='phytochemMiner',
    version='1.0',
    packages=find_packages(include=['phytochemMiner']),
    install_requires=[
        'langchain_core',
        'pydantic_core',
        'pandas',
        'tqdm',
        'typing',
        'langchain_deepseek'
    ],
    url='https://github.com/alrichardbollans/miningphytochemicals',
    license='Attribution-NonCommercial-ShareAlike 4.0 International',
    author='Adam Richard-Bollans',
    description='A package for extracting mentions of phytochemicals from scientific articles.',
    long_description=open('readme.md', encoding="utf8").read()
)
