from datetime import datetime
from functools import partial

import pandas as pd
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import pyqtSignal
from spikeinterface.curation import train_model
import numpy as np

class TrainWindow(QtWidgets.QMainWindow):

    update_signal = pyqtSignal()

    def __init__(self, project):

        super().__init__()

        project_folder = project.folder_name

        train_model_kwargs = {}

        window_title_text = "UNITREFINE: Train your model"
        self.setWindowTitle(window_title_text)

        self.model_folder = None

        labels = []
        metrics_paths = []
        metrics = []
        analyzers_for_training = []
        for analyzer in project.analyzers.values():
            analyzer_folder = project.folder_name / analyzer['analyzer_in_project']
            
            labels_path = analyzer_folder / "labels.csv"
            if labels_path.is_file():

                labels_df = pd.read_csv(labels_path).sort_values('unit_id')
                labels.append(labels_df['quality'].values)

                metrics_path = analyzer_folder / "labelled_metrics.csv"
                metrics_paths.append(metrics_path)
                metrics.append(pd.read_csv(metrics_path))

                analyzers_for_training.append(analyzer)

        train_model_kwargs['metrics_paths'] = metrics_paths
        metric_names_set = set()
        
        for metric_list in metrics:
            metric_names_for_one_sa = set(metric_list.columns)
            if len(metric_names_set) == 0:
                metric_names_set = metric_names_for_one_sa
            else:
                metric_names_set = metric_names_set.intersection(metric_names_for_one_sa)

        train_model_kwargs['labels'] = labels
        parent_folder = project_folder / 'models' 

        metric_names = [metric_name for metric_name in metric_names_set if metric_name not in ["index", "quality", "unit_id"]]
        train_model_kwargs['metric_names'] = metric_names

        data_text = "Using the following analyzers:<br />"
        for analyzer_dict, metric_data in zip(analyzers_for_training, metrics, strict=True):

            data_text += f"{analyzer_dict['path']}: {len(metric_data)} units curated.<br />"

        data_text += f"<br />Metrics shared by all analyzer are: {metric_names}."

        formLayout = QtWidgets.QFormLayout()
        widget = QtWidgets.QWidget()
        widget.setStyleSheet("background-color: PeachPuff")
        label_1 = QtWidgets.QTextEdit(f"<h3>Information</h3><p>Here, you can train many models based on the labelled data.</p>{data_text}")
        label_1.setReadOnly(True)
        label_1.setStyleSheet("background-color: white")

        blank_label = QtWidgets.QLabel("")

        self.classifiersForm = QtWidgets.QLineEdit("['RandomForestClassifier']")
        self.classifiersForm.setStyleSheet("background-color: white")
        classifiersOptions = QtWidgets.QLabel("<i>Possible options</i>: 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'SVC',<br /> 'LogisticRegression', 'XGBClassifier', 'CatBoostClassifier', 'LGBMClassifier', 'MLPClassifier'.")

        self.scalarsForm = QtWidgets.QLineEdit("['standard_scaler']")
        self.scalarsForm.setStyleSheet("background-color: white")

        scalarsOptions = QtWidgets.QLabel("<i>Possible options</i>: 'standard_scaler', 'min_max_scaler', 'robust_scaler'")

        self.imputersForm = QtWidgets.QLineEdit("['knn']")
        self.imputersForm.setStyleSheet("background-color: white")
        imputersOptions = QtWidgets.QLabel("<i>Possible options</i>: 'median', 'most_frequent', 'knn', 'iterative'")

        self.testSizeForm = QtWidgets.QLineEdit("0.2")
        self.testSizeForm.setStyleSheet("background-color: white")

        trainButton = QtWidgets.QPushButton("Train models")
        trainButton.clicked.connect(partial(self.do_training, train_model_kwargs, parent_folder, project))

        codeButton = QtWidgets.QPushButton("Generate code to train a model")
        codeButton.clicked.connect(partial(self.generate_code, train_model_kwargs, parent_folder))

        #trainButton.setStyleSheet("")

        formLayout.addRow(label_1)
        formLayout.addRow(blank_label)

        formLayout.addRow("List of Classifiers: ", self.classifiersForm)
        formLayout.addRow(classifiersOptions)
        formLayout.addRow(blank_label)

        formLayout.addRow("List of Scalars: ", self.scalarsForm)
        formLayout.addRow(scalarsOptions)
        formLayout.addRow(blank_label)

        formLayout.addRow("List of Imputers: ", self.imputersForm)
        formLayout.addRow(imputersOptions)
        formLayout.addRow(blank_label)

        formLayout.addRow("Test size: ", self.testSizeForm)
        formLayout.addRow(blank_label)

        formLayout.addRow(trainButton)
        formLayout.addRow(codeButton)

        widget.setLayout(formLayout)

        self.setCentralWidget(widget)

    def generate_code(self, train_model_kwargs, parent_folder):

        model_folder = parent_folder / f'model_{datetime.now():%Y-%m-%d_%H:%M:%S}'

        code_text = "\n# Here is the code being executed by the 'Train Model' button above\n"
        code_text += "# Feel free to play with the different arguments!\n\n"
        code_text += "import spikeinterface.full as si\n\n"
        code_text += f"model = si.train_model(\n    mode='csv',\n     imputation_strategies={eval(self.imputersForm.text())},\n    scaling_techniques={eval(self.scalarsForm.text())},\n    classifiers={eval(self.classifiersForm.text())},\n    test_size={eval(self.testSizeForm.text())},\n    folder={model_folder}\n"
        for key, value in train_model_kwargs.items():
            code_text += f"    {key} = {value},\n"
        code_text += ")\n\n"
        code_text += f"# Your model is saved at {model_folder}\n\n"
        code_text += "Read more here: https://spikeinterface.readthedocs.io/en/stable/tutorials/curation/plot_2_train_a_model.html\n\n"      

        print(code_text)


    def do_training(self, train_model_kwargs, parent_folder, project):

        imputation_strategies = eval(self.imputersForm.text())
        scaling_techniques = eval(self.scalarsForm.text())
        classifiers = eval(self.classifiersForm.text())
        test_size = eval(self.testSizeForm.text())

        model_folders_in_project = project.models
        if len(model_folders_in_project) > 0:
            model_indices = np.array([int(str(model_path.name).split('__')[-1]) for model_path in model_folders_in_project])
            max_model_index = np.max(model_indices) if len(model_indices) > 0 else 0
        else:
            max_model_index = 0

        folder = parent_folder / f'model__{max_model_index + 1}'  

        print("\nUsing UnitRefine, inside SpikeInterface, to train models...\n")

        train_model(
            mode="csv",
            imputation_strategies=imputation_strategies,
            scaling_techniques=scaling_techniques,
            classifiers=classifiers,
            test_size=test_size,
            folder=folder,
            **train_model_kwargs,
        )

        project.models.append(folder)

        print(f"\nFinished training models! Best model saved in in {folder}.\n")

        self.model_folder = folder


    def closeEvent(self, event):
        """Intercepts the user closing the app, to save the labels."""

        self.update_signal.emit()
        #self.parent_window.update_signal.emit()
        event.accept()
