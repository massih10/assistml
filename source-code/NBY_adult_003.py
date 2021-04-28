import sys
import json
import multiprocessing
import openml
import logging
import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models, create_model, tune_model, finalize_model, predict_model, \
    save_model, load_model
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import decomposition

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class BuildMLModel:
    """
    Class to build the ML model
    This class needs to be more generalized.
    """

    def __init__(self):
        """
        Constructor
        Doing nothing for now. Need to be more cool later
        """
        self.dataset = None
        self.X = None
        self.y = None
        self.categorical_indicator = None
        self.clf = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_preprocessed = None
        self.X_test_preprocessed = None
        self.X_feature_names = None
        self.X_preprocessed_feature_names = None
        self.y_feature_name = None
        self.pycaret_setup_obj = None

    def download_dataset(self, dataset_id):
        """ Download the dateset with id from openml
            :param   dataset_id - id of the dataset to be downloaded from openml

            :returns    tupple of  (X, y, categorical_indicator, attribute_names, dataset), where
                        X - features
                        y - the classes for each example
                        categorical_indicator - an array that indicates which feature is categorical
                        attribute_names - the names of the features for the examples (X) and target feature (y)
                        dataset - openml dataset object
        """
        self.dataset = openml.datasets.get_dataset(dataset_id)

        self.X, self.y, self.categorical_indicator, self.X_feature_names = self.dataset.get_data(
            dataset_format='dataframe',
            target=self.dataset.default_target_attribute
        )
        logger.info(self.X.head)
        logger.info(self.X.describe())
        logger.info("Shape of the X = {}".format(self.X.shape))
        logger.info("Is Categorical? = {}".format(self.categorical_indicator))

        self.y_feature_name = self.y.name

        return self.X, self.y, self.categorical_indicator, self.X_feature_names, self.dataset

    def split_dataset(self, train_size, random_state=42, sampling=None):
        """Split the dataset into train-test of the train_size size
                :param  X: full feature matrix
                :param  y:the classes for each dataset in X
                :param train_size: train/test split ratio
                :param random_state: seed value for the split
                :param sampling: sampling type ("stratified" or None)
                :returns tupple of (X_train, X_test, y_train, y_test), where
                         X_train: dataset of features for training
                         X_test: dataset of features for testing
                         y_train: class labels for training
                         y_test: class labels for testing
            """
        if "stratified" in sampling.lower():
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                    train_size=train_size,
                                                                                    random_state=random_state,
                                                                                    stratify=self.y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                    train_size=train_size,
                                                                                    random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def convert_pandas_series_to_dataframe(self, data):
        """
        Converts pandas series type to dataframe type

        :param data: pandas series type data
        :return: dataframe type
        """
        return pd.DataFrame(data)

    def store_train_test_dataset_in_csv(self, train_csv_name, test_csv_name):
        """
        Stores data-sets with labels in csv file

        :param X_train: Features of training dataset
        :param X_test: Features of testing dataset
        :param y_train: Labels of training dataset
        :param y_test: Labels of testing dataset
        :param train_csv_name: train dataset csv filename
        :param test_csv_name: test dataset csv filename
        :return: None
        """
        # convert the y labels series to dataframe to be stored in csv
        y_train_df = self.convert_pandas_series_to_dataframe(self.y_train)
        y_test_df = self.convert_pandas_series_to_dataframe(self.y_test)

        training_frames = [self.X_train, y_train_df]
        test_frames = [self.X_test, y_test_df]
        test_frames = [self.X_test_preprocessed, y_test_df]
        print('----------------------------------')
        print(test_frames[1].columns)
        print(self.X_test_preprocessed)
        print(self.y_test)
        print('----------------------------------')
        train_df = pd.concat(training_frames, axis=1)
        test_df = pd.concat(test_frames, axis=1)

        # remove the existing files
        if os.path.exists(train_csv_name):
            os.remove(train_csv_name)
        if os.path.exists(test_csv_name):
            os.remove(test_csv_name)

        train_df.to_csv(train_csv_name, index=True, header=True)
        test_df.to_csv(test_csv_name, index=True, header=True)

        logger.info("Train {0} and Test {1} files are available".format(train_csv_name, test_csv_name))

    def data_pre_processing(self):
        """
        Performs pre-processing of data-sets
        :return: tupple of (X_train_processed, X_test_processed), where
                 X_train_processed: processed X_train
                 X_test_processed: processed X_test
        """

        logger.info("Missing Values - \n{0}".format(self.X.isnull().sum()))

        # There are no missing values. So no need to do this ---------
        # self.X_train_preprocessed = self.X.dropna()
        # logger.info("Shape after dropping missing data rows = {0}".format(self.X_train_preprocessed.shape))
        # logger.info("It dropped {} number of rows".format(self.X.shape[0] - self.X_train_preprocessed.shape[0]))
        # Convert the missing values to 'most_frequent'
        # logger.info("Doing Simple Imputer based on 'most_frequent'")
        # imp_mean = impute.SimpleImputer(strategy='most_frequent')
        # self.X_train_preprocessed = imp_mean.fit_transform(self.X_train)
        # self.X_test_preprocessed = imp_mean.fit_transform(self.X_test)

        # Get the categorical columns
        logger.info("Extracting categorical columns only")
        categorical_feature_names = []
        for index, is_nominal in enumerate(self.categorical_indicator):
            if is_nominal:
                categorical_feature_names.append(self.X_feature_names[index])
        X_train_categorical = self.X_train[categorical_feature_names]
        X_test_categorical = self.X_test[categorical_feature_names]
        logger.info("categorical columns {}".format(categorical_feature_names))
        logger.info("Shape of X_train with only categorical columns {}".format(X_train_categorical.shape))
        logger.info("Shape of X_test with only categorical columns {}".format(X_test_categorical.shape))

        # Get the non categorical columns
        logger.info("Extracting non categorical columns only")
        non_categorical_feature_names = []
        for index, is_nominal in enumerate(self.categorical_indicator):
            if not is_nominal:
                non_categorical_feature_names.append(self.X_feature_names[index])
        X_train_numeric = self.X_train[non_categorical_feature_names]
        X_test_numeric = self.X_test[non_categorical_feature_names]
        logger.info("Non categorical columns {}".format(non_categorical_feature_names))
        logger.info("Shape of X_train with only non categorical columns {}".format(X_train_numeric.shape))
        logger.info("Shape of X_test with only non categorical columns {}".format(X_test_numeric.shape))

        # Convert all the nominal features to Label Encoder
        for index, feature in enumerate(categorical_feature_names):
            col_le = preprocessing.LabelEncoder()
            logger.info("Convert column '{}' to LabelEncoder".format(feature))
            X_train_categorical.iloc[:, index] = col_le.fit_transform(X_train_categorical.iloc[:, index].to_list())
            X_test_categorical.iloc[:, index] = col_le.transform(X_test_categorical.iloc[:, index].to_list())
            logger.info("Column '{}' transformed to LabelEncoder with classes {}".format(feature, col_le.classes_))

        # Transforming the categorical y label to LabelEncoder
        le = preprocessing.LabelEncoder()
        self.y_train = le.fit_transform(self.y_train)
        self.y_test = le.transform(self.y_test)
        logger.info("target label transformed to LabelEncoder with classes {}".format(le.classes_))

        # Do chi2 feature selection for 75% of the categorical features
        n_components = int(0.75 * X_train_categorical.shape[1])
        logger.info("Performing chi square best {} feature selections on categorical features".format(n_components))
        select_k_features = feature_selection.SelectKBest(feature_selection.chi2, k=n_components)
        X_train_categorical_preprocessed = select_k_features.fit_transform(X_train_categorical, self.y_train)
        X_test_categorical_preprocessed = select_k_features.transform(X_test_categorical)

        k_features_indices = select_k_features.get_support(True)

        logger.info(
            "Number of columns after chi square feature selection = {}".format(
                X_train_categorical_preprocessed.shape[1]))
        logger.info("{} best feature indices are {}".format(n_components, k_features_indices))
        X_categorical_preprocessed_feature_names = []
        for index in k_features_indices:
            X_categorical_preprocessed_feature_names.append(categorical_feature_names[index])

        logger.info("Categorical features selected are {}".format(X_categorical_preprocessed_feature_names))

        # Convert into dataframe
        X_train_categorical_preprocessed = pd.DataFrame(X_train_categorical_preprocessed,
                                                        columns=X_categorical_preprocessed_feature_names)
        X_test_categorical_preprocessed = pd.DataFrame(X_test_categorical_preprocessed,
                                                       columns=X_categorical_preprocessed_feature_names)

        # Do RobustScaler for numircal data
        logger.info("Performing Robust Scaling on numerical data")
        rs = preprocessing.RobustScaler()
        X_train_numeric = rs.fit_transform(X_train_numeric)
        X_test_numeric = rs.transform(X_test_numeric)

        X_train_numeric = pd.DataFrame(X_train_numeric, columns=non_categorical_feature_names)
        X_test_numeric = pd.DataFrame(X_test_numeric, columns=non_categorical_feature_names)

        # Do PCA for 75% of the numeric features
        n_components = int(0.75 * X_train_numeric.shape[1])
        pca = decomposition.PCA(n_components=n_components)
        pca.fit_transform(X_train_numeric)  # project the original data into the PCA space

        logger.info("No. of PCA axis = {}".format(len(pca.explained_variance_ratio_)))
        logger.info("Total variance = {}".format(pca.explained_variance_ratio_.sum()))
        logger.info("PCA Components = {}".format(pca.components_))
        logger.info("Dropping {} and {} from non categorical features.".format(non_categorical_feature_names[0],
                                                                               non_categorical_feature_names[2]))
        X_train_numeric_preprocessed = X_train_numeric.drop(columns=[non_categorical_feature_names[0],
                                                                     non_categorical_feature_names[2]])
        X_test_numeric_preprocessed = X_test_numeric.drop(columns=[non_categorical_feature_names[0],
                                                                   non_categorical_feature_names[2]])

        X_non_categorical_preprocessed_feature_names = []
        for index, col in enumerate(non_categorical_feature_names):
            if index != 0 and index != 2:
                X_non_categorical_preprocessed_feature_names.append(col)

        logger.info("Non Categorical features selected are {}".format(X_non_categorical_preprocessed_feature_names))

        self.X_train_preprocessed = pd.concat([X_train_categorical_preprocessed,
                                               X_train_numeric_preprocessed],
                                              axis=1)
        self.X_test_preprocessed = pd.concat([X_test_categorical_preprocessed,
                                              X_test_numeric_preprocessed],
                                             axis=1)
        self.X_preprocessed_feature_names = X_categorical_preprocessed_feature_names + X_non_categorical_preprocessed_feature_names
        logger.info("Best features are = {}".format(self.X_preprocessed_feature_names))

        return self.X_preprocessed_feature_names, self.X_train_preprocessed, self.X_test_preprocessed, \
               X_train_numeric_preprocessed, X_train_categorical_preprocessed

    def setup_pycaret(self, seed_value):
        y_train_df = self.convert_pandas_series_to_dataframe(self.y_train)
        # y_test_df = self.convert_pandas_series_to_dataframe(self.y_test)

        y_train_df.columns = [self.y_feature_name]
        # y_test_df.columns = [self.y_feature_name]

        training_frames = [self.X_train_preprocessed, y_train_df]
        # test_frames = [self.X_test_preprocessed, y_test_df]
        train_df = pd.concat(training_frames, axis=1)
        # test_df = pd.concat(test_frames, axis=1)

        # setup pycaret
        self.pycaret_setup_obj = setup(data=train_df, target=self.y_feature_name, session_id=seed_value,
                                       data_split_shuffle=True, data_split_stratify=True)
        return self.pycaret_setup_obj

    def compare_all_models(self):
        best_model = None
        if self.pycaret_setup_obj is not None:
            best_model = compare_models()
        return best_model

    def create_NGB_classifier(self):
        """
        Creates and returns a Naive Gaussian Bayes Classifier
        :return: clf: Naive Gaussian Bayes Classifier
        """
        self.clf = create_model('nb')
        return self.clf

    def tune_model(self, params):
        tune_start_time = time.time()
        self.clf = tune_model(self.clf, custom_grid=params)
        tune_stop_time = time.time()
        duration = tune_stop_time - tune_start_time
        logger.info("Classifier tuned in {0} seconds".format(duration))
        return self.clf, duration

    def fit_with_classifier(self):
        """
        Fit the training data to classifier and record the duration of training in seconds
        """
        train_start_time = time.time()
        self.clf = finalize_model(self.clf)
        train_stop_time = time.time()

        duration = train_stop_time - train_start_time
        logger.info("Classifier fit done in {0} seconds".format(duration))
        return self.clf, duration

    def predict_with_classifier(self):
        """
        Predict using classifier
        """
        y_test_df = self.convert_pandas_series_to_dataframe(self.y_test)
        y_test_df.columns = [self.y_feature_name]
        test_frames = [self.X_test_preprocessed, y_test_df]
        test_df = pd.concat(test_frames, axis=1)

        test_start_time = time.time()
        unseen_predictions = predict_model(self.clf, data=test_df)
        test_stop_time = time.time()

        duration = test_stop_time - test_start_time
        logger.info("Prediction done in {0} seconds".format(duration))

        y_pred = unseen_predictions[[self.y_feature_name]].to_numpy()
        y_pred = np.array([pred[0] for pred in y_pred])

        if isinstance(y_pred[0], str):
            y_pred = y_pred.astype(np.int)  # also convert the string type classes to int

        return y_pred, duration

    def performance_metric(self, y_pred_top_1, y_pred_top_n, precision_recall_average):
        """
        Calculates accuracy, precision, recall and confusion matrix
        :param y_pred_top_1: predicted labels of top 1 class
        :param y_pred_top_n: predicted labels of top n classes
        :param precision_recall_average: average for calculating the precision, recall
        :return: tupple of (overall_accuracy_top_1, overall_accuracy_top_n, overall_precision, overall_recall, overall_fscore, conf_matrix), where
                overall_accuracy_top_1: accuracy
                overall_precision: precision
                overall_recall: recall
                overall_fscore: f score
                conf_matrix: confusion matrix
        """
        # convert the y_test categorical dtype to numpy array for easy comparison
        y_test_series = pd.Series(self.y_test)
        y_test = y_test_series.to_numpy().astype(np.int)  # also convert the string type classes to int

        overall_accuracy_top_1 = accuracy_score(y_test, y_pred_top_1)

        overall_accuracy_top_n = None
        if y_pred_top_n is not None:
            overall_accuracy_top_n = np.mean(
                np.array([1 if int(y_test[k]) in y_pred_top_n[k] else 0 for k in range(len(y_test))]))

        # 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
        overall_precision, overall_recall, overall_fscore, _ = precision_recall_fscore_support(y_test, y_pred_top_1,
                                                                                               average=precision_recall_average)

        logger.info("Overall accuracy for top 1 -> {0}".format(overall_accuracy_top_1))
        if overall_accuracy_top_n is not None:
            logger.info("Overall accuracy for top n -> {0}".format(overall_accuracy_top_n))
        logger.info("Overall precision -> {0}".format(overall_precision))
        logger.info("Overall recall -> {0}".format(overall_recall))

        # Rows indicate true labels and column indicate predicted labels
        conf_matrix = confusion_matrix(y_test, y_pred_top_1)
        logger.info("Confusion matrix - {0}".format(conf_matrix))

        if overall_accuracy_top_n is not None:
            return overall_accuracy_top_1, overall_accuracy_top_n, overall_precision, overall_recall, overall_fscore, conf_matrix
        return overall_accuracy_top_1, overall_precision, overall_recall, overall_fscore, conf_matrix

    def save_model_pickle(self, pickle_file_name, pickle_file_base_path):
        """
        Persist the model in the pickle file
        :param clf: classifier to be persisted
        :param pickle_file_name: Name of the pickle file
        :param pickle_file_base_path: Path to the pickle dir
        :return:
        """
        pickle_file_full_path = os.path.join(pickle_file_base_path, pickle_file_name)
        save_model(self.clf, pickle_file_full_path)
        logger.info("Wrote to pickle file {0}.pkl".format(pickle_file_full_path))

    def retrieve_model_pickle(self, pickle_file_name, pickle_file_base_path, replace=False):
        """
        Retrieve the model from the pickle file
        :param pickle_file_name: Name of the pickle file
        :param pickle_file_base_path: Path to the pickle dir
        :return: clf: the retrieved classifier
        """
        pickle_file_full_path = os.path.join(pickle_file_base_path, pickle_file_name)
        imported_model = None
        if os.path.exists(pickle_file_full_path + '.pkl'):
            imported_model = load_model(pickle_file_full_path)

            if replace:
                self.clf = imported_model

            logger.info("Pickle file loaded {0}.pkl".format(pickle_file_full_path))
        else:
            raise FileNotFoundError("The {0}.pkl pickle file does not exists".format(pickle_file_full_path))

        return imported_model


class CreateJsonTree:
    """
    Class to create JSON Tree
    """

    def __init__(self):
        pass

    @staticmethod
    def create_model_node(info_node, data_meta_data_node, training_characteristics_node, metrics_node):
        """
            Build the model node in json file
            :param info_node: Info node dictionary
            :param data_meta_data_node: data_meta_data node dictionary
            :param training_characteristics_node: training_characteristics node dictionary
            :param metrics_node: Metric node dictionary
            :return: model_parent_node: dictionary of Model node
        """

        model_node = {}
        model_node["Info"] = info_node
        model_node["Data_Meta_Data"] = data_meta_data_node
        model_node["Training_Characteristics"] = training_characteristics_node
        model_node["Metrics"] = metrics_node
        model_parent_node = {"Model": model_node}
        return model_parent_node

    @staticmethod
    def create_data_meta_data_node(dataset_name, rows, cols_pre_preprocessing, classification_type,
                                   class_variable, classification_output,
                                   output_length, categorical_columns, numeric_columns, datetime_columns, text_columns,
                                   list_of_columns_used,
                                   cols_afr_preprocessing, preprocessing_dict):
        """
        Build the data_meta_data node in json file.
        :param dataset_name: dataset name
        :param rows: No. of samples
        :param cols_pre_preprocessing: No. of attributes/features  before pre-processing
        :param classification_type: Single-class or multi-class classification
        :param class_variable: The target class
        :param classification_output: single or prediction_probabilities
        :param output_length: No. of prediction_probabilities in the output ( top n prediction)
        :param categorical_columns: No. of categorical columns
        :param numeric_columns: No. of numerical columns
        :param datetime_columns: No. of data time columns
        :param text_columns: No. of text columns
        :param list_of_columns_used: List of all the attributes used
        :param cols_afr_preprocessing: No. of attributes/features  after pre-processing
        :param preprocessing_dict: All pre-processing param dictionary
        :return: data_meta_data_node: dictionary of all the Data_Meta_Data node
        """
        data_meta_data_node = {}

        data_meta_data_node["dataset_name"] = dataset_name
        data_meta_data_node["rows"] = rows
        data_meta_data_node["cols_pre_preprocessing"] = cols_pre_preprocessing
        data_meta_data_node["classification_type"] = classification_type
        data_meta_data_node["class_variable"] = class_variable
        data_meta_data_node["classification_output"] = classification_output
        data_meta_data_node["output_length"] = output_length

        data_meta_data_node["categorical_columns"] = categorical_columns
        data_meta_data_node["numeric_columns"] = numeric_columns
        data_meta_data_node["datetime_columns"] = datetime_columns
        data_meta_data_node["text_columns"] = text_columns
        data_meta_data_node["list_of_columns_used"] = list_of_columns_used
        data_meta_data_node["cols_afr_preprocessing"] = cols_afr_preprocessing

        data_meta_data_node["Preprocessing_node"] = preprocessing_dict

        return data_meta_data_node

    @staticmethod
    def create_training_characteristics_node(hyper_parameters_dict, test_size, seed_value, cross_validation_folds,
                                             sampling,
                                             algorithm_implementation, language, language_version, cores, ghZ,
                                             deployment, implementation,
                                             dependencies_dict):
        """
        Build the training_characteristics node in json file.
        :param hyper_parameters_dict: Model hyperparameter dictionary
        :param test_size: split test size
        :param seed_value: seed value used for the split
        :param cross_validation_folds: No. of cross validation folds
        :param sampling: Train test split was stratified or not
        :param algorithm_implementation: Classifier used for training
        :param language: Programming language used
        :param language_version: version of the language used
        :param cores: No. of cores of the CPU
        :param ghZ: Speed of the CPU useddependencies_dict
        :param deployment: single_host or multi_hosts
        :param implementation: single_language or multi_languages
        :param dependencies_dict: dictionary of dependencies
        :return: training_characteristics_node: dictionary of all thetraining_characteristics node
        """

        training_characteristics_node = {}

        training_characteristics_node["Hyper_Parameters"] = hyper_parameters_dict
        training_characteristics_node["test_size"] = test_size
        training_characteristics_node["seed_value"] = seed_value
        training_characteristics_node["cross_validation_folds"] = cross_validation_folds
        training_characteristics_node["sampling"] = sampling
        training_characteristics_node["algorithm_implementation"] = algorithm_implementation
        training_characteristics_node["language"] = language
        training_characteristics_node["language_version"] = language_version
        training_characteristics_node["cores"] = cores
        training_characteristics_node["ghZ"] = ghZ
        training_characteristics_node["deployment"] = deployment
        training_characteristics_node["implementation"] = implementation
        training_characteristics_node["Dependencies"] = dependencies_dict

        return training_characteristics_node

    @staticmethod
    def create_metrics_node(overall_accuracy_top_1, overall_accuracy_top_n, overall_precision, overall_recall,
                            overall_fscore,
                            training_time, cross_validated_training_time, test_time_per_unit, conf_matrix, test_file,
                            avg_type):
        """
        Build the metrics node in json file.
        :param overall_accuracy_top_1: Top 1 accuracy rate
        :param overall_accuracy_top_n: Top N accuracy rate
        :param overall_precision: Overall precision calculated used the average type avg_type
        :param overall_recall: Overall recall calculated used the average type avg_type
        :param overall_fscore: Overall fscore calculated used the average type avg_type
        :param training_time: Duration of the training in seconds
        :param cross_validated_training_time: Duration of the cross validation in seconds
        :param test_time_per_unit: Duration of the each test prediction in seconds
        :param conf_matrix: Confusion matrix
        :param test_file: Test CSV file name
        :param avg_type: Average type used tp calculate precision, recall and fscore
        :return: metrics_node: dictionary of all the metrics node
        """
        metrics_node = {}

        if overall_accuracy_top_1 is not None:
            metrics_node["accuracy"] = overall_accuracy_top_1

        if overall_accuracy_top_n is not None:
            metrics_node["accuracy_top_n"] = overall_accuracy_top_n

        if overall_precision is not None:
            metrics_node["precision"] = overall_precision

        if overall_recall is not None:
            metrics_node["recall"] = overall_recall

        if overall_fscore is not None:
            metrics_node["fscore"] = overall_fscore

        if training_time is not None:
            metrics_node["training_time"] = training_time

        if cross_validated_training_time is not None:
            metrics_node["cross_validated_training_time"] = cross_validated_training_time

        if test_time_per_unit is not None:
            metrics_node["test_time_per_unit"] = test_time_per_unit

        if conf_matrix is not None:
            metrics_node["confusion_matrix_rowstrue_colspred"] = conf_matrix

        if test_file is not None:
            metrics_node["test_file"] = test_file

        if avg_type is not None:
            metrics_node["avg_type"] = avg_type

        return metrics_node

    @staticmethod
    def create_info_node(task_name, spec_version, use_case):
        """

        :param task_name: name of the task
        :param spec_version: Version
        :param use_case: Use case
        :return: info_node: dictionary of all the info node
        """

        info_node = {}

        info_node["name"] = task_name
        info_node["spec_version"] = spec_version
        info_node["use_case"] = use_case
        return info_node


class JsonFileReadWrite:
    json_file_name = None
    json_file_base_path = None
    json_file_full_path = None

    def __init__(self, json_file_name, json_file_base_path):
        self.json_file_name = json_file_name
        self.json_file_base_path = json_file_base_path
        self.json_file_full_path = os.path.join(self.json_file_base_path, self.json_file_name)

    def write_json_file(self, data):
        """
        Write into json file
        :param data: python dictionary to write into the json file
        :return: None
        """
        if not isinstance(data, dict):
            raise TypeError("Json data is not in python dictionary format")

        # remove the existing files
        if os.path.exists(self.json_file_full_path):
            os.remove(self.json_file_full_path)

        with open(self.json_file_full_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

        logger.info("Created JSON file at {0}".format(self.json_file_full_path))

    def read_json_file(self):
        """
        Read from json file
        :return: data: Json data in form of python dictionary
        """
        data = None
        with open(self.json_file_full_path) as json_file:
            data = json.load(json_file)

        if data is not None:
            data = dict(data)

        logger.info("Read from JSON file at {0}".format(self.json_file_full_path))

        return data


if __name__ == "__main__":
    #os.chdir("../")
    path = '/home/mohammoa/HiWi/asm-2'
    data_base_path = os.path.join(path, "1_data")
    pickle_file_base_path = os.path.join(path, "3_pickle")
    json_file_base_path = os.path.join(path, "3_pickle")
    os.chdir(os.path.join(path, "2_code"))

    # Info starts--------------------------------------
    task_name = "NBY_adult_003"
    spec_version = "1.0.2"
    use_case = "adult"
    info_node_dict = CreateJsonTree.create_info_node(task_name, spec_version, use_case)

    # Data_Meta_Data starts--------------------------------------
    dataset_id = 1590
    dataset_name = "adult"
    build_ml_model_obj = BuildMLModel()

    # download the bank dataset (id 1461)
    X, y, categorical_indicator, attribute_names, dataset_openml = build_ml_model_obj.download_dataset(dataset_id)
    class_variable = build_ml_model_obj.y_feature_name

    classification_type = ""
    if "pandas.core.series.Series" in str(type(y)):
        n_classes = len(y.unique())
        if n_classes > 2:
            classification_type = "Multi-Class"
        else:
            classification_type = "Single-Class"

    output_length = 1
    if output_length == 1:
        classification_output = "single"
    else:
        classification_output = "prediction_probabilities"

    # first split the data and then do the pre-processing
    # ------------------------------------------------------------------------------------------------------
    # This part belongs to Training_Characteristics node but we need to split and store the data in csv files
    # before pre-processing
    train_size = 0.6
    test_size = 1 - train_size
    seed_value = 42
    cross_validation_folds = 10
    sampling = "Stratified"
    stratify_var = class_variable

    # split the train-test data
    X_train, X_test, y_train, y_test = build_ml_model_obj.split_dataset(train_size, seed_value, sampling)

    sampling = "Stratified on {0}".format(stratify_var)

    train_csv_name = "adult_train_" + str(int(train_size * 100)) + ".csv"
    train_csv_path = os.path.join(data_base_path, train_csv_name)
    test_csv_name = "adult_test_" + str(int((1 - train_size) * 100)) + ".csv"
    test_csv_path = os.path.join(data_base_path, test_csv_name)


    # do pre-processing
    feature_names, X_train_preprocessed, X_test_preprocessed, \
    X_train_numeric_preprocessed, X_train_categorical_preprocessed = build_ml_model_obj.data_pre_processing()

    # store the train-test data in csv
    build_ml_model_obj.store_train_test_dataset_in_csv(train_csv_path, test_csv_path)
    # ------------------------------------------------------------------------------------------------------


    # Create the Data_Meta_Data node for replication process
    rows = X_train_preprocessed.shape[0] + X_test_preprocessed.shape[0]
    cols_pre_preprocessing = X.shape[1]
    categorical_columns = X_train_categorical_preprocessed.shape[1]
    numeric_columns = X_train_numeric_preprocessed.shape[1]
    datetime_columns = 0
    text_columns = 0
    list_of_columns_used = feature_names
    cols_afr_preprocessing = X_train_preprocessed.shape[1]

    preprocessing_dict = {}
    preprocessing_dict["categorical_encoding"] = "LabelEncoding"
    preprocessing_dict["numerical_encoding"] = "RobustScaler"
    preprocessing_dict["numeric_selection"] = "Best 75% features using PCA"
    preprocessing_dict["categorical_selection"] = "Best 75% features using chi2"
    preprocessing_dict["date_encoding"] = "None"
    preprocessing_dict["text_encoding"] = "None"
    data_meta_data_node_dict = CreateJsonTree.create_data_meta_data_node(dataset_name, rows, cols_pre_preprocessing,
                                                                         classification_type,
                                                                         class_variable, classification_output,
                                                                         output_length,
                                                                         categorical_columns, numeric_columns,
                                                                         datetime_columns,
                                                                         text_columns, list_of_columns_used,
                                                                         cols_afr_preprocessing,
                                                                         preprocessing_dict)

    # Training_Characteristics starts--------------------------------------

    # create classifier - Naive Gaussian Bayes Classifier
    algorithm_implementation = "pycaret.classification.create_model('nb')"

    pycaret_setup_obj = build_ml_model_obj.setup_pycaret(seed_value)
    NGB_clf = build_ml_model_obj.create_NGB_classifier()
    logger.info("Default NGB model {}".format(NGB_clf))

    priors_list = None
    var_smoothing_list = [-4e-09, -2e-09, 2e-09, 4e-09, 6e-09, 8e-09, 10e-09, 20e-09, 40e-09, 60e-09, 80e-09, 100e-09]

    # find out a good var_smoothing hyperparameter
    params = {'var_smoothing': var_smoothing_list}
    tuned_NGB_clf, tune_duration = build_ml_model_obj.tune_model(params=params)
    logger.info("After tuning NGB model {}".format(tuned_NGB_clf))

    hyper_parameters_dict = {}
    grid_search_dict = {}
    if priors_list is not None:
        grid_search_dict["priors"] = priors_list

    if var_smoothing_list is not None:
        grid_search_dict["var_smoothing"] = [str(v) for v in var_smoothing_list]
    hyper_parameters_dict["Grid_Search"] = grid_search_dict

    #var_smoothing = -2e-09
    var_smoothing = tuned_NGB_clf.var_smoothing
    hyper_parameters_dict["var_smoothing"] = str(var_smoothing)

    language = "python"
    language_version = sys.version.split()[0]
    cores = multiprocessing.cpu_count()
    ghZ = "1.70"
    deployment = "single_host"
    implementation = "single_language"

    precision_recall_average = "macro"

    dependencies_dict = {}
    platforms_dict = {}
    platforms_dict["pycharm"] = "2020.3.1 (Community Edition)"
    platforms_dict["pip"] = "20.3.3"
    platforms_dict["scikit-learn"] = "0.23.2"
    platforms_dict["pycaret"] = "2.2.3"
    dependencies_dict["Platforms"] = platforms_dict
    libraries_dict = {}
    libraries_dict["openml"] = "0.10.2"
    libraries_dict["pandas"] = "1.1.2"
    libraries_dict["numpy"] = "1.19.2"
    libraries_dict["json"] = "1.19.2"

    dependencies_dict["Libraries"] = libraries_dict
    dependencies_dict["nr_dependencies"] = len(platforms_dict) + len(libraries_dict)
    training_characteristics_node_dict = CreateJsonTree.create_training_characteristics_node(hyper_parameters_dict,
                                                                                             test_size,
                                                                                             seed_value,
                                                                                             cross_validation_folds,
                                                                                             sampling,
                                                                                             algorithm_implementation,
                                                                                             language, language_version,
                                                                                             cores, ghZ,
                                                                                             deployment, implementation,
                                                                                             dependencies_dict)

    # Metrics starts--------------------------------------
    # fit the classifier to training data-set
    final_NGB_clf, train_duration = build_ml_model_obj.fit_with_classifier()

    # store the model into pickle file
    pickle_file_name = "NBY_adult_003"  # No need to put .pkl extension. pycaret save model automatically puts .pkl extension
    build_ml_model_obj.save_model_pickle(pickle_file_name, pickle_file_base_path)

    # load the model from the pickle file
    clf_retrieved = build_ml_model_obj.retrieve_model_pickle(pickle_file_name, pickle_file_base_path, replace=True)

    # get the prediction from the classifier
    y_pred, test_duration = build_ml_model_obj.predict_with_classifier()

    test_time_per_unit = test_duration / y_test.shape[0]

    # calculate the performance metrics
    overall_accuracy_top_1, overall_precision, overall_recall, overall_fscore, conf_matrix = build_ml_model_obj.performance_metric(
        y_pred, None, precision_recall_average)
    overall_accuracy_top_n = None

    # convert the ndarray to make it json serializable
    if 'numpy.ndarray' in str(type(conf_matrix)):
        conf_matrix = conf_matrix.tolist()

    metric_node_dict = CreateJsonTree.create_metrics_node(overall_accuracy_top_1=overall_accuracy_top_1,
                                                          overall_accuracy_top_n=overall_accuracy_top_n,
                                                          overall_precision=overall_precision,
                                                          overall_recall=overall_recall,
                                                          overall_fscore=overall_fscore,
                                                          training_time=train_duration,
                                                          cross_validated_training_time=None,
                                                          test_time_per_unit=test_time_per_unit,
                                                          conf_matrix=conf_matrix,
                                                          test_file=test_csv_name, avg_type=precision_recall_average)

    model_node_dict = CreateJsonTree.create_model_node(info_node=info_node_dict,
                                                       data_meta_data_node=data_meta_data_node_dict,
                                                       training_characteristics_node=training_characteristics_node_dict,
                                                       metrics_node=metric_node_dict)

    json_file_name = "NBY_adult_003.json"
    jsonWriteObj = JsonFileReadWrite(json_file_name, json_file_base_path)
    jsonWriteObj.write_json_file(model_node_dict)
