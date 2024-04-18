import os
import re
from glob import glob
from os.path import basename, splitext
import setuptools

def fetch_files_from_folder(folder):
    options = glob(f"{folder}/**", recursive=True)
    data_files = []
    # All files inside the folder need to be added to package_data
    # which would include yaml configs as well as project READMEs
    for option in options:
        if os.path.isdir(option):
            files = []
            for f in glob(os.path.join(option, "*")):
                if os.path.isfile(f):
                    files.append(f)
                data_files += files
    return data_files

def fetch_package_data():
    current_dir = os.getcwd()
    tsgv_folder = os.path.dirname(os.path.abspath(__file__))
    # The files for package data need to be relative to mmf package dir
    os.chdir(os.path.join(tsgv_folder, "code"))
    data_files = fetch_files_from_folder(".")
    os.chdir(current_dir)
    return data_files

if __name__ == "__main__": 
        
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="MoLEF",
        version="1.0.0",
        author="MoLEF",
        author_email="author@example.com",
        description="Moment localization evaluation framework",
        long_description=long_description,
        long_description_content_type="text/markdown",

        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        package_dir={"": "code"},
        packages=setuptools.find_packages(where='code'),
        py_modules=[splitext(basename(path))[0] for path in glob('code/*.py')], 
        python_requires=">=3.8",
    )
