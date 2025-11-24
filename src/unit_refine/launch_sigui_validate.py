import argparse
import functools
import sys
from pathlib import Path

import pandas as pd
import spikeinterface.full as si
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QWidget
from pyqtgraph import mkQApp
from spikeinterface_gui.backend_qt import QtMainWindow
from spikeinterface_gui.controller import Controller


def my_custom_close_handler(event: QCloseEvent, window: QWidget, project_folder, save_folder, analyzer):
    """
    This function will be called instead of the original closeEvent.
    """
    re_labelled_unit_ids = []
    re_labels = []

    for row in curation_dict['manual_labels']:
        
        if row.get('labels') is None:
            if row.get('quality') is not None:
                re_labels.append(row['quality'][0])
                re_labelled_unit_ids.append(int(row['unit_id']))
        else:
            if row.get('labels').get('quality') is not None:
                re_labels.append(row['labels']['quality'][0])
                re_labelled_unit_ids.append(int(row['unit_id']))

    labels_df = pd.DataFrame()
    labels_df['quality'] = re_labels
    labels_df['unit_id'] = re_labelled_unit_ids
    labels_df = labels_df.sort_values('unit_id')

    labels_df.to_csv(save_folder / "relabelled_units.csv", index=False)


argv = sys.argv[1:]

parser = argparse.ArgumentParser(description='spikeinterface-gui')
parser.add_argument('analyzer_folder', help='SortingAnalyzer folder path', default=None, nargs='?')
parser.add_argument('project_folder', help='Project folder path', default=None, nargs='?')
parser.add_argument('analyzer_in_project', help='Project folder path', default=None, nargs='?')
parser.add_argument('model_predictions_file')

args = parser.parse_args(argv)

analyzer_folder = Path(args.analyzer_folder)
project_folder = Path(args.project_folder)
analyzer_in_project = Path(args.analyzer_in_project)
model_predictions_file = Path(args.model_predictions_file)

save_folder = project_folder / analyzer_in_project
save_folder.mkdir(exist_ok=True, parents=True)

model_decisions = pd.read_csv(model_predictions_file)

analyzer = si.load_sorting_analyzer(analyzer_folder, load_extensions=False)

manual_labels = []
for unit_id in analyzer.unit_ids:
    decision = {"unit_id": unit_id, 
                "model": model_decisions[model_decisions['unit_id'] == unit_id]['prediction'].values,
                }
    if len(decision['model']) > 0:
        manual_labels.append(decision)
    

label_definitions = {
    "quality": dict(name="quality", label_options=["good", "MUA", "noise"], exclusive=True),
    "model": dict(name="model", label_options=["good", "MUA", "noise"], exclusive=True),
}

extra_unit_properties = {'confidence': model_decisions['probability'].values}

curation_dict = dict(
    format_version="2",
    unit_ids=analyzer.unit_ids,
    manual_labels=manual_labels,
    label_definitions=label_definitions,
)

controller = Controller(
    analyzer,
    backend="qt",
    curation=True,
    curation_data=curation_dict,
    extra_unit_properties=extra_unit_properties,
    displayed_unit_properties=['model', 'quality', 'confidence', 'firing_rate', 'snr', 'x', 'y', 'rp_violations']
)

layout_dict={'zone1': ['unitlist'], 'zone2': [], 'zone3': ['waveform'], 'zone4': ['correlogram'], 'zone5': ['spikeamplitude'], 'zone6': [], 'zone7': [], 'zone8': ['spikerate']}

app = mkQApp()
win = QtMainWindow(controller, layout_dict=layout_dict, user_settings=None)
win.closeEvent = functools.partial(my_custom_close_handler, window=win, project_folder=project_folder, save_folder=save_folder, analyzer=analyzer)

win.show()
app.exec()
