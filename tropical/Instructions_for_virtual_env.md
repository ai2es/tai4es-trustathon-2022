# Instructions for setting up the `ai2es` virtual environment

### Note: These instructions are only applicable if you are trying to run any of the Trustathon notebooks locally on your own machine.  If you are using the Trustathon JupyterHub, the default kernel should have all of the dependencies installed and should run conflict-free.

To make sure that all of the packages in `notebook-tropical.ipnyb` can run conflict-free, it is important to set up and run the `ai2es` virtual environment.  Virtual environments freeze package versions and can manage conflicts.  This is especially important for packages that sometimes have issues with backwards compatibility (looking at you, `tensorflow`).  <b>Please note that if you do not run our notebook in the `ai2es` virtual environment, it is quite likely you will run into version conflicts with `tensorflow` and some of the other packages!</b>. 

I will provide instructions for using `Miniconda` to manage your environments.  `Miniconda` is not the only option for doing this and there may be better choices out there; however, if you have strong preferences about how to manage your virtual environments, you probably don't need these instructions!

## 1. Creating the virtual environment
Navigate to the `tai4es-trustathon-2022` directory.  You should see a file called `environment.yml` in this directory--this is the file that will give your machine the instructions for how to construct the `ai2es` virtual environment.  After you have located `environment.yml`, simply run the following command:

`conda env create -f environment.yml`

<b>NOTE</b>: this can take a pretty long time to run, particularly the `solving environment` step.  So don't panic if it's going slowly!

## 2. Activate environment
Activate the `ai2es` virtual environment by running:

`conda activate ai2es`

You should now see that you are in the `ai2es` environment rather than your base environment.  Your command line should now look something like this:

`(ai2es) [user@machine tai4es-trustathon-2022]$ `

If you need to change environments for some reason, you can exit the `ai2es` environment using the `deactivate` comand:

`conda deactivate`

## 3. (Optional) Install `ipykernel`
<b>NOTE: depending on how you have Jupyter configured, you may not need to do this step!</b>

Ultimately, you want to be able to run `notebook-tropical.ipynb` in the `ai2es` environment.  This means that when you open the notebook, you should be able to see that you are running in `ai2es`, not your base environment `Python3`.  If you have activated `ai2es` before opening your Jupyter notebook, you *should* be running in the `ai2es` environment; however, depending on your Jupyter configuration, you might need to complete one final step.  So if you still seem to be running in the base `Python3` environment, read on.

Install `ipykernel`:  After making sure you have activated the `ai2es` environment, install `ipykernel`:

`conda install ipykernel`

Then, make sure Jupyter/ipython will be able to see the `ai2es` environment:

`ipython kernel install --user --name=ai2es`

This should allow you to run your notebooks in the `ai2es` environment.  (Note that in addition to assigning your environment in the command line, you can also switch environments once you have opened a Jupyter notebook by selecting `Kernel` from the dropdown menu, selecting `Change kernel`, and choosing `ai2es`).  

