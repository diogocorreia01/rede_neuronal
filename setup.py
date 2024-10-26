from setuptools import setup, find_packages

setup(
    name='rede_neuronal',
    version='1.0.0',
    description='Rede neuronal desenvolvida para o projeto1 da cadeira de IASC',
    author='Diogo Correia',
    author_email='diogo.f.correia@protonmail.com',
    url='https://github.com/diogocorreia01/rede_neuronal.git',
    packages=find_packages(),
    install_requires=[
        'absl-py==2.1.0',
        'astunparse==1.6.3',
        'certifi==2024.8.30',
        'charset-normalizer==3.4.0',
        'flatbuffers==24.3.25',
        'gast==0.6.0',
        'google-pasta==0.2.0',
        'grpcio==1.67.0',
        'h5py==3.12.1',
        'idna==3.10',
        'keras==3.6.0',
        'libclang==18.1.1',
        'Markdown==3.7',
        'markdown-it-py==3.0.0',
        'MarkupSafe==3.0.2',
        'mdurl==0.1.2',
        'ml-dtypes==0.4.1',
        'namex==0.0.8',
        'numpy==2.0.2',
        'opt_einsum==3.4.0',
        'optree==0.13.0',
        'packaging==24.1',
        'pillow==11.0.0',
        'protobuf==5.28.3',
        'Pygments==2.18.0',
        'requests==2.32.3',
        'rich==13.9.3',
        'setuptools==75.2.0',
        'six==1.16.0',
        'tensorboard==2.18.0',
        'tensorboard-data-server==0.7.2',
        'tensorflow==2.18.0',
        'termcolor==2.5.0',
        'typing_extensions==4.12.2',
        'urllib3==2.2.3',
        'Werkzeug==3.0.6',
        'wheel==0.44.0',
        'wrapt==1.16.0'
    ],
    python_requires='>=3.12',
)