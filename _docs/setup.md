---
title: Setup
layout: default
---

[Setup]({{brainsmith.github.io}}/docs/setup/)
|
[Tutorial]({{brainsmith.github.io}}/docs/tutorial/)
|
[CLI]({{brainsmith.github.io}}/docs/usage/)

---

# Steps to install Brainsmith

There were many dependency issues faced at the time of the compiler installation due to deprecated libraries and modules or dependency onto older versions of the python libraries and missing dependencies.

1. Git clone the repo:

       git clone https://github.com/microsoft/brainsmith.git ./brainsmith
       cd brainsmith/
2. Create the python environment and activate using the command:

       virtualenv -p python3.11 .venv && sed -i 's/PS1="(.*/PS1="($(basename $PWD)-py3.11) $PS1"/' .venv/bin/activate
3. For some of the dependencies that we found uninstalled in our local system, such as tkinter, pyverilator, we made sure these were also installed, using the pip command

       pip install pyverilator
       pip install tkinter

4. The main dependency issue was faced with the FINN Compiler installtion, and its dependencies. This issue is mainly attributed to the installed of a python library called '_setuptools_'. Its version could be either out of reach (>68.0.0) or a degraded one (<65.0.0).
   Following this, we made sure to check and install this library, under the '_finn-experimental_' directory.

       (brainsmith-py3.11) kirthanar@KirthanaR:~$ cd brainsmith/deps/finn-experimental
       (brainsmith-py3.11) kirthanar@KirthanaR:~$ pip install --no-build-isolation -e .
   
   To install setuptools contaning the '_pkg_resources_' module.

       pip install --no-cache-dir --force-reinstall --no-build-isolation --use-pep517 setuptools==65.5.1 wheel poetry
   To verify the successful installation of the library:

       python -c "import pkg_resources; print('Fixed:', pkg_resources.__version__)"

   ### OR

       (brainsmith-py3.11) kirthanar@KirthanaR:~$ cd brainsmith/
       (brainsmith-py3.11) kirthanar@KirthanaR:~/brainsmith$ poetry run pip install "setuptools==68.2.2" --force-reinstall && poetry run pip install --no-build-isolation -e ./deps/finn-experimental && poetry run python -c "import finn.experimental; print('READY ’)"
       pip install wheel
   and comment out the following line, in the pyproject.toml file, which is located within this directory.

       finn-experimental = {path = "deps/finn-experimental", develop = true, optional = false}
   
6. Now, the shell script can be executed to resume the Brainsmith compiler installation, using the '_force_' keyword:

       (brainsmith-py3.11) kirthanar@KirthanaR:~/brainsmith$ ./setup-venv.sh --force

   Incase of an error again, w.r.t the '_pkg_resources_' library, run the followng command, under the brainsmith directory:
   
       poetry lock
       poetry install
       poetry run pip install -e deps/finn-experimental --no-use-pep517
   And check the installation status again using the same verification command. Following this, we've run the shell setup script once gain.
   The below images, are a result of the second installation method.

   <img width="721" height="403" alt="image" src="https://github.com/user-attachments/assets/755ba493-4c4f-42ee-9d20-c82da1755b03" />

   <img width="773" height="436" alt="image" src="https://github.com/user-attachments/assets/c65a6b2d-dfc7-4590-a62e-949a07242187" />
