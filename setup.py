from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='pysparse',
    version='0.1.0',
    description='Package to encode and decode large OOM numpy arrays as Sparse binaries',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(),
    author='Thomas Frost',
    author_email='tdgfrost@gmail.com',
    keywords=['PySparse', 'SparseArray'],
    url='https://github.com/tdgfrost/PySparse',
    download_url=''
)

install_requires = [
    'numpy>=1.18.0',
    'tqdm>=4.46.0',
    'numba>=0.49.0',
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
