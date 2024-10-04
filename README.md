# D2C (Dependency to Causality)
**D2C (Dependency to Causality)** is a library for causal discovery. It focuses on computing asymmetric conditional mutual information terms, known as descriptors, within the estimated Markov blankets of variable pairs.

N.B. This procedure was tested in Linux. If you have Windows, you can use WSL.

## Clone 
Make sure you only get the latest commit 
```
git clone --depth 1 https://github.com/gmpal/d2c
```

If you already have a working environment, go to step [2](#step2)

## Python, poetry
To get started with TD2C, follow these steps:

1. Install everything you need for pyenv
    ```
    sudo apt update
    sudo apt install -y build-essential libssl-dev libffi-dev python3-dev zlib1g-dev libbz2-dev libsqlite3-dev libreadline-dev libgdbm-dev liblzma-dev tk-dev
    ```

2. Install `pyenv` (recommended, tested):
        ```
        curl https://pyenv.run | bash
        ```

2. Load pyenv automatically by appending the following to ~/.bash_profile if it exists, otherwise ~/.profile (for login shells) and  ~/.bashrc (for interactive shells)
     
    ```
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
        ```
3. Restart the shell. Then 

    ```
    pyenv install 3.10
    pyenv virtualenv 3.10 td2c
    pyenv shell td2c
    ```

4. Install Poetry for dependencies management:
    ```
    pip install --upgrade pip
    pip install poetry
    ```

5. Install dependencies from `poetry.lock` file:
    ```
    poetry env use $(pyenv which python)
    poetry install
    ```

6. If the above fails, you can resolve dependencies and try again. This will take a few minutes. 
    ```
    poetry lock
    poetry install
    ```

## Environment already setup <a name="step2"></a>
    ```
    pip install poetry
    poetry install
    ```

If you already have an environment, you can simply run from the project directory
5. Setup completed. Now you can run a jupyter notebook and access the notebooks provided. 
    
    ```
    poetry shell
    jupyter notebook
    ```


## Usage
Check the notebooks in the [notebooks folder](./notebooks) for detailed examples and explanations:

- [00_data_generation.ipynb](./notebooks/00_data_generation.ipynb): Demonstrates how to generate synthetic data for testing the TD2C library.
- [01_descriptors_computation.ipynb](./notebooks/01_descriptors_computation.ipynb): Shows how to compute the descriptors using the TD2C library.
- [02_performances.ipynb](./notebooks/02_performances.ipynb): Provides results of cross validation performances.
