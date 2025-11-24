# UnitRefine

[UnitRefine: A Community Toolbox for Automated Spike Sorting Curation](https://www.biorxiv.org/content/10.1101/2025.03.30.645770v1)

You can use UnitRefine to manually label sorted units (using [SpikeInterface-GUI](https://github.com/SpikeInterface/spikeinterface-gui/) as a backend), then train a model based on this labelled to curate for you! If you already have a model, you can use UnitRefine to validate that it does what you expect.

You'll need a SortingAnalyzer from [SpikeInterface](https://github.com/SpikeInterface/spikeinterface-gui/) to use it. It's [pretty easy](https://spikeinterface.readthedocs.io/en/stable/how_to/build_pipeline_with_dicts.html) to make one!

## Installation

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/), the modern python package manager.

1. Clone this repository and move into the repo folder

``` bash
git clone https://github.com/chrishalcrow/unit_refine.git
cd unit_refine
```

2. Open unit_refine, creating a new project

``` bash
uv run unit_refine --project_folder my_new_project
```

Note: you must be in the unit_refine folder that you've cloned from github when you run this command.

A window should pop up that looks something like this:

![image](resources/basic_window.png)

From here, it should be easy to add sorting analyzers, curate the data, train a model and validate your model. Keep an eye on the feedback that comes through the terminal - it will help! You can also generate code which you could use in a Python script.

Whenever you curate something or make a model, whatever you've done is automatically saved in your project folder. Next time you run ``unit_refine``, just point to your existing folder and it will load:

``` bash
uv run unit_refine --project_folder my_existing_project
```

## Thanks

UnitRefine is highly dependent on the flexible and powerful SpikeInterface and Spikeinterface-GUI packages. Many thanks to Alessio, Sam, Zack, Joe who gave help and feedback to this project, and to the entire SpikeInterface team <3
