from setuptools import setup, find_packages


with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='pysparse-array',
    version='1.0.0',
    description='Package to encode and decode large OOM numpy arrays as Sparse binaries',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    packages=find_packages(),
    author='Thomas Frost',
    author_email='tdgfrost@gmail.com',
    url='https://github.com/tdgfrost/PySparse',
    download_url='https://pypi.org/project/pysparse-array/'
)

if __name__ == '__main__':
    setup(**setup_args)
