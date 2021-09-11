# Running Topcoffea with Work Queue

The script `work_queue_run.py` sets up topcoffea to run as a Work Queue
application. Work Queue itself is a framework for building large scale
applications. With Work Queue, `work_queue_run.py` serves as a manager that
waits for worker processes to connect, and dispatches to them work to complete.

When using Work Queue, it is your responsibility to launch the worker processes
according to your needs, for example, using a campus cluster, or a
supercomputer on XSEDE. You should also setup your python environment so that
it can easily be sent to the remote workers. We will cover these two points
below.


## Obtaining topcoffea

We highly recommend setting up topcoffea as git repository. This allows
topcoffea to automatically detect changes that need to be included in the
python environments sent to the workers:

```sh
git clone https://github.com/TopEFT/topcoffea.git
cd topcoffea
```


## Setting up python

The recommended way to set up `topcoffea` with Work Queue is using the `conda`
python package manager. If you do not have `miniconda` (recommended, as it
installs much faster), or `anaconda` installed, you must run the following
steps on a terminal:

```sh
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```

Once `conda` is installed, open a new terminal and create the base python
environment for topcoffea:

```sh
# you may choose other python version, e.g. 3.9
conda create --name topcoffea-env python=3.8
conda activate topcoffea-env
conda install -c conda-forge coffea xrootd ndcctools dill conda-pack

# install topcoffea via pip. We install it in editable mode to ease the test of
# changes in development. From the root directory of the topcoffea repository:
pip install -e .

# You may install any other modules that you are developing, as:
# cd /path/to/my/module
# pip install -e .
```

---
**NOTE**

If your python environments do not work after the step `conda activate
topcoffea-env`, for example, if `python` immediately fails because it cannot
find module, then your conda installation may be in conflict with a previous
setup. Most problems like this are solved by typing:

```sh
unset PYTHONPATH
```

just before the `conda activate` command.

---

This completes setting up the base python environment. The python environment
sent to the workers will be automatically constructed from the base environment
as changes in the topcoffea code are detected. We highly recommend that when
installing topcoffea, `pip install -e .` is executed from a git repository.
This allows `work_queue_run.py` to detect unstaged changes or newer commits and
rebuild the environment accordingly.


## Executing the topcoffea application

The script `work_queue_run.py` expects a configuration file that describes
which files of events to process. One small configuration to test is:


```sh
conda activate topcoffea-env
cd analysis/topEFT

## optional: initialize your proxy credentials to access the needed xrootd files.
## It is not needed if the .cfg file is using root files from local paths.
# voms-init-proxy2

python work_queue_run.py --chunksize 128000 ../../topcoffea/cfg/mc_signal_samples.cfg
```

The first time you run `work_queue_run.py` it will spend a handful of minutes
constructing the environment that will be sent to the workers. After that, it
will wait for workers to connect.


## Launching workers

Work Queue has several mechanisms to execute worker processes. The simplest way
is to launch a local worker for testing purposes. `work_queue_run.py` creates a
Work Queue manager named after your user id, and the worker uses this name to
find the address of the manager. In some other terminal, run:

```sh
conda activate topcoffea-env
work_queue_worker -dall --cores 1 --memory 8000 --disk 8000 -M ${USER}-workqueue-coffea
```

We use the options cores, memory, and disk to limit the resources that worker
claims for itself, otherwise it may incorrectly assume that the whole resources
of the machine are available to it. Further, the resources used here match the
resource description in `work_queue_run.py` for the maximum resources any task
can use. When no task has been completed, tasks are dispatched to workers using
these maximum resources specified, so it is required for the workers to be at
least as large.  As tasks are finished and data about their resource usage is
collected, Work Queue will try to reduce the size of the allocations used when
first trying the tasks in order to run more tasks concurrently. Tasks will be
retried, if needed, using the maximum specified values in `work_queue_run.py`.

We also use the option `-dall` to print debugging output. If the worker
correctly connected to the manager, you should see the messages that describe
the work submited to the worker.

In a similar way, we can launch workers using a campus cluster that has HTCondor:

```sh
conda activate topcoffea-env
condor_submit_workers --cores 4 --memory 16000 --disk 16000 -M ${USER}-workqueue-coffea 10
```

In this case, we are submitting 10 workers, each with 4 cores, and 4GB of
memory and disk per core.

Instead of launching the workers manually, we can use the `work_queue_factory`
to submit workers for us. The factory checks with the manager if there is work
available, and submit workers accordingly. We specify the arguments for the
factory in a configuration file:

factory.json
```json
{
    "manager-name": "USER-workqueue-coffea",
    "max-workers": 10,
    "min-workers": 0,
    "workers-per-cycle": 10,
    "cores": 4,
    "memory": 16000,
    "disk": 16000,
    "timeout": 900,
}
```


```sh
# Remember to replace USER in the manager-name of the configuration file with
# your user id.
conda activate topcoffea-env
work_queue_factory -Tcondor -Cfactory.json
```

The greatest advantage of using a configuration file for the factory is that
when this file is updated, the factory reconfigures itself. This is very useful
when controlling the minimum and maximum number of workers.


## Exploring the executor arguments

The script `work_queue_run.py` has some other default options that may be
tweaked to run particular workflows more efficiently, or that generate debug
output. These are documented in the `executor_args` dictionary in
`work_queue_run.py`. For example, statistics of the run are sent to a file
called `stats.log`, which can be plotted using:

```sh
conda activate topcoffea-env
work_queue_graph_log -Tpng stats.log
```

