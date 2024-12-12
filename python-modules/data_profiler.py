import pandas as pd
import math
from scipy import stats
import json
import numpy as np
import pymongo
import datetime
import sys
import base64
import io
import time
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import collections
import scipy
from sklearn.feature_selection import f_classif, mutual_info_classif
from functools import reduce



class DataProfiler():

    ##########################################################################################################
    ########################################## Function Definition ###########################################
    ##########################################################################################################

    def __init__(self, mode, dataset_path, df, dataset_name, target_label, target_feature_type, use_case,
                 feature_annotation_list):
        self.mode = mode
        self.dataset_path = dataset_path
        self.df = df
        self.dataset_name = dataset_name
        self.class_label = target_label
        self.target_feature_type = target_feature_type
        self.use_case = use_case
        self.nr_total_features = 0
        self.feature_annotation_list = feature_annotation_list
        self.csv_data = ''
        self.csv_data_complete = ''
        self.column_names_list = ''
        self.miss_value = ''
        self.drop_cols = ''
        self.json_data = {}
        self.numerical_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.unstructured_features = []
        self.collection_datasets = ''
        self.json_data["Info"] = {}
        self.json_data["Features"] = {}
        self.json_data["Info"]["dataset_name"] = self.dataset_name
        self.json_data["Info"]["use_case"] = use_case
        self.json_data["Info"]["target_label"] = self.class_label
        self.json_data["Info"]["target_feature_type"] = self.target_feature_type

    # Return a new dataset after dropping missing values. And return Number of missing values in each column.
    def handle_missing_values(self, df):
        print('size of df before dropping missing values: ' + str(len(df)))
        print("Number of missing values in each column:")
        miss_value = (df.isnull().sum())
        print(miss_value)
        null_stats = miss_value.sum()
        print("Number of missings data in the whole dataset: " + str(null_stats))

        # drop the column if this col has missing values more than half of the whole dataset
        drop_cols = list()
        num_rows = len(df)
        for index, val in enumerate(miss_value):
            if val > num_rows / 4:
                drop_cols.append(df.columns[index])
        print("Dropped columns: " + str(drop_cols))
        df = df.drop(drop_cols, axis=1)

        # Drop the rows even with single NaN or single missing values.
        df = df.dropna()
        print('size of df after dropping missing values: ' + str(len(df)))
        return (df, miss_value, drop_cols)

        # Calculate Interquartile range and Quartiles

    def iqr_cal(self, df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q2 = df_in[col_name].quantile(0.5)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        q0 = q1 - (1.5 * iqr)
        q4 = q3 + (1.5 * iqr)
        return q0, q1, q2, q3, q4, iqr

    # Detect the outliers in a column using interquartile range
    def detect_outlier(self, df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        # print("fence_low: "+str(fence_low)+", fence_high: "+str(fence_high))
        df_out = df_in[col_name].loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)]
        return df_out.to_numpy()

    # Return correlation between second argument(target col) and current column
    def corr_cal(self, df, col_name):
        return (df[col_name].corr(df[self.class_label]))

    # Return minimum order in current column
    def min_orderm_cal(self, df, col_name):
        if abs(df[col_name].min()) == 0:
            return -math.inf
        else:
            return (round(math.log10(abs(df[col_name].min()))))

    # Return maximum order in current column
    def max_orderm_cal(self, df, col_name):
        if abs(df[col_name].max()) == 0:
            return -math.inf
        else:
            return (round(math.log10(abs(df[col_name].max()))))

    # Return number of categories/levels in a column for categorical feature
    def freq_counts(self, df, col_name):
        res = df[col_name].value_counts()
        val_list = res.tolist()
        index_list = res.index.tolist()
        return index_list, val_list, len(index_list)

    # Retun how big is the imbalance of feature (ratio between most popular and least popular)
    def imbalance_test(self, df, col_name):
        res = df[col_name].value_counts().tolist()
        return max(res) / min(res)

    # Chi-square Test of Independence using scipy.stats.chi2_contingency
    # The H0 (Null Hypothesis): There is no relationship between variable one and variable two.
    # Null hypothesis is rejected when the p-value is less than 0.05
    def chisq_correlated_cal(self, df, col_name):
        crosstab = pd.crosstab(df[self.class_label], df[col_name])
        res = stats.chi2_contingency(crosstab)
        p = res[1]
        # return p-value, We can reject the null hypothesis as the p-value is less than 0.05
        if p < 0.05:
            ifCorr = 'True'
        else:
            ifCorr = 'False'
        return (p, ifCorr)

    def text_statistics(self, df, col_name):
        feature = df[col_name]
        min_vocab = 1000
        max_vocab = 0
        vocab_size = 0
        print("total_text")
        new_words = []
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        #print(stopwords.words())
        #print(type(stopwords.words()))
        #exit()

        for i, txt in enumerate(feature):
            text_tokens = word_tokenize(txt)
            vocab_size_doc = len(text_tokens)
            vocab_size += vocab_size_doc
            if vocab_size_doc > max_vocab:
                max_vocab = vocab_size_doc
            if vocab_size_doc < min_vocab:
                min_vocab = vocab_size_doc
            new_words += tokenizer.tokenize(txt)
            #new_words = list(set(new_words))


        print("tokens_without_sw")
        print(len(new_words))
        tokens_without_sw = list(reduce(lambda x,y : filter(lambda z: z!=y,x) ,stopwords.words('english'),new_words))
        # now = datetime.datetime.now()
        # for i, word in enumerate(new_words):
        #     if word not in stopwords.words():
        #         tokens_without_sw.append(word)
        #     if i/len(new_words) >= 0.1:
        #         then = datetime.datetime.now()
        #         total = then-now
        #         print(i)
        #         print('estimated remaining time: ', total, ' seconds')
        #tokens_without_sw = [word for word in new_words if not word in stopwords.words()]

        print("relative vocabulary")
        # relative vocabulary
        nm = len(tokens_without_sw)
        relative_vocab = vocab_size / nm

        print("vocabulary concentration")
        # vocabulary concentration
        elements_count = collections.Counter(tokens_without_sw)
        vocab_freq_lst = sorted(tokens_without_sw, key=lambda x: -elements_count[x])
        vocab_freq_lst = pd.unique(vocab_freq_lst).tolist()
        top_10 = []
        for i in range(10):
            top_10.append(vocab_freq_lst[i])
        n_top = 0
        for i, x in enumerate(top_10):
            n_top += tokens_without_sw.count(x)
        vocab_concentration = n_top / len(tokens_without_sw)

        print("entropy")
        # entropy
        data = pd.Series(tokens_without_sw)
        counts = data.value_counts()
        entropy = scipy.stats.entropy(counts)

        return vocab_size, relative_vocab, vocab_concentration, entropy, min_vocab, max_vocab

    # claculate the monotonous filtering for the numerical features
    def monotonous_filtering_numerical(self, df, col_name):
        feature = df[col_name]
        mean = feature.mean(axis=0)
        std = feature.std(axis=0)
        fence_1 = mean - std
        fence_2 = mean + std
        total_number_of_values = len(feature)
        number_of_features_inside_fences = 0
        for idx, i in enumerate(feature):
            if fence_1 <= i <= fence_2:
                number_of_features_inside_fences += 1
        percentage_of_monotonic_values = number_of_features_inside_fences / total_number_of_values
        return percentage_of_monotonic_values

    # claculate the monotonous filtering for the categorical features
    def monotonous_filtering_categorical(self, df, col_name):
        feature = df[col_name]
        levels = []
        frequency = []
        for idx, i in enumerate(feature):
            if i not in levels:
                levels.append(i)
                frequency.append(1)
            else:
                frequency[levels.index(i)] += 1
        num_highest_levels = math.ceil(0.1 * len(frequency))
        highest_levels = []
        freq_level = zip(frequency, levels)
        freq_level_sorted = sorted(freq_level, reverse=True)
        for i in range(num_highest_levels):
            highest_levels.append(freq_level_sorted[i][1])
        total_number_of_values = len(feature)
        number_of_values_in_highest_levels = 0
        for idx, i in enumerate(feature):
            if i in highest_levels:
                number_of_values_in_highest_levels += 1
        percentage_of_monotonic_values = number_of_values_in_highest_levels / total_number_of_values
        return percentage_of_monotonic_values

    # Shapiro-Wilk test for normality.
    # H0 (Null Hypothesis): Normal distributed.
    # p value less than 0.05 means that null hypothesis is rejected.
    def shapiro_test_normality(self, df, col_name):
        shapiro_test = stats.shapiro(df[col_name])
        if shapiro_test[1] < 0.05:
            return False
        else:
            return True

    # Kolmogorov Smirnov test for Exponential distribution.
    # H0 (Null Hypothesis): Exponentially distributed.
    # p value less than 0.05 means that null hypothesis is rejected.
    def ks_test_exponential(self, df, col_name):
        ks_test = stats.kstest(df[col_name], 'expon')
        if ks_test.pvalue < 0.05:
            return False
        else:
            return True

    # Converts numpy datatypes into python default datatypes
    def datatype_converter(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    # Check if information about this dataset is already available in DB and return status in boolean.
    def check_data_availability_in_db(self, dataset_info_json):
        myclient = pymongo.MongoClient("mongodb://admin:admin@localhost:27017/")
        try:
            # List all databases
            db_list = myclient.list_database_names()
            print("Connection successful. Databases available:")
            print(db_list)
        except pymongo.errors.ConnectionError as e:
            print("Connection failed:", e)
            
        dbname = myclient["assistml"]
        self.collection_datasets = dbname["datasets"]
        similar_document = self.collection_datasets.find(
            {'Info.dataset_name': dataset_info_json["Info"]['dataset_name'],
             'Info.use_case': dataset_info_json["Info"]['use_case'],
             'Info.features': dataset_info_json["Info"]['features'],
             'Info.numeric_ratio': dataset_info_json["Info"]['numeric_ratio'],
             'Info.categorical_ratio': dataset_info_json["Info"]['categorical_ratio'],
             'Info.datetime_ratio': dataset_info_json["Info"]['datetime_ratio'],
             'Info.unstructured_ratio': dataset_info_json["Info"]['unstructured_ratio']}
            , {"_id": 0})
        similar_docs = list(similar_document)  # Convert cursor to list
        print(similar_docs)
        if (len(similar_docs)== 0):
            return False
        else:
            return True

    # Read pandas dataframe and handle missing values
    def process_pandas_df(self):
        missing_values = ["n/a", "na", "--", "NA", "?"," ?", "", " ", "NAN", "NaN"]
        if self.mode == 1:
            self.csv_data = pd.read_csv(self.dataset_path + "/" + self.dataset_name, sep=",", na_values=missing_values)
        elif self.mode == 2:
            decoded = base64.b64decode(self.df)
            self.csv_data = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep=",", na_values=missing_values)
        self.column_names_list = list(self.csv_data.columns)

        self.csv_data_complete = pd.DataFrame()
        self.csv_data_complete = pd.concat([self.csv_data_complete, self.csv_data], axis=1)

        # Check if the target label provided by the user exists
        if not self.class_label in self.column_names_list:
            return "processing failed"
        # handle_missing_values
        (self.csv_data, self.miss_value, self.drop_cols) = self.handle_missing_values(self.csv_data)
        self.json_data["Info"]["observations"] = self.csv_data.shape[0]
        self.nr_total_features = self.csv_data.shape[1] - 1  # Do not count class label
        self.json_data["Info"]["features"] = self.nr_total_features
        return ("processing success")

    # Identify indices of numerical, categorical features
    def process_feature_annotation_list(self):
        self.feature_annotation_list = self.feature_annotation_list.replace(' ', '')
        self.feature_annotation_list = self.feature_annotation_list.replace("'", '')
        self.feature_annotation_list = self.feature_annotation_list.replace('"', '')
        feature_types_list = list(self.feature_annotation_list.strip('[]').split(','))
        # Identify indices of numerical and categorical features
        for i in range(0, len(feature_types_list)):
            if feature_types_list[i] == 'N':
                self.numerical_features.append(i)
            elif feature_types_list[i] == 'C':
                self.categorical_features.append(i)
            elif feature_types_list[i] == 'D':
                self.datetime_features.append(i)
            elif feature_types_list[i] == 'U':
                self.unstructured_features.append(i)
            elif feature_types_list[i] == 'T':
                print("Identified Class Label in annotation list")
            else:
                print("Please recheck the feature annotation list")
                column_list = self.column_names_list
                for column in self.drop_cols:
                    column_list.remove(column)
                return column_list[i]
                # exit(1)
        return "parsing successfully completed"

    # Calculate ratios
    def calculate_ratios(self):
        # Calculate ratios
        nr_numeric_features = len(self.numerical_features)
        nr_categorical_features = len(self.categorical_features)
        nr_datetime_features = len(self.datetime_features)
        nr_unstructured_features = len(self.unstructured_features)
        self.json_data["Info"]["numeric_ratio"] = float("{:.2f}".format(nr_numeric_features / self.nr_total_features))
        self.json_data["Info"]["categorical_ratio"] = float(
            "{:.2f}".format(nr_categorical_features / self.nr_total_features))
        self.json_data["Info"]["datetime_ratio"] = float("{:.2f}".format(nr_datetime_features / self.nr_total_features))
        self.json_data["Info"]["unstructured_ratio"] = float("{:.2f}".format(nr_unstructured_features / self.nr_total_features))

    # Calculate parameters for numerical features and add it to json
    def analyse_numerical_features(self):
        print("Analysing numerical features")
        # Calculate parameters for numerical features and add it to json
        self.json_data["Features"]["Numerical_Features"] = {}
        self.json_data["Info"]["analyzed_features"] = []
        self.json_data["Info"]["discarded_features"] = []
        feature = ""
        numericalFeatures = pd.DataFrame()
        for i in range(len(self.numerical_features)):
            numericalFeatures = pd.concat([numericalFeatures, self.csv_data[self.csv_data_complete.columns[self.numerical_features[i]]]],
                                         axis=1)
        if not (len(numericalFeatures) == 0):
            anova_f1 = f_classif(numericalFeatures, self.csv_data[self.class_label])[0]
            anova_pvalue = f_classif(numericalFeatures, self.csv_data[self.class_label])[1]
            if 'categoric' in self.target_feature_type or 'Categoric' in self.target_feature_type or 'binary' in self.target_feature_type or 'Binary' in self.target_feature_type or 'categorical' in self.target_feature_type or 'Categorical' in self.target_feature_type:
                mi = mutual_info_classif(numericalFeatures, self.csv_data[self.class_label])
            else:
                mi = mutual_info_classif(numericalFeatures, self.csv_data[self.class_label])
        counter = 0
        try:
            for column_nr in self.numerical_features:
                feature = self.column_names_list[column_nr]
                if (feature not in self.drop_cols):
                    self.json_data["Info"]["analyzed_features"].append(feature)
                    self.json_data["Features"]["Numerical_Features"][feature] = {}
                    # Implement the monotonous filtering
                    self.json_data["Features"]["Numerical_Features"][feature]['monotonous_filtering'] = self.monotonous_filtering_numerical(self.csv_data, feature)
                    # Assign the f1 value from the anova:
                    self.json_data["Features"]["Numerical_Features"][feature]['anova_f1'] = anova_f1[counter]
                    # Assign the p value from the anova:
                    self.json_data["Features"]["Numerical_Features"][feature]['anova_pvalue'] = anova_pvalue[counter]
                    # Assign the mutual information for the feature
                    self.json_data["Features"]["Numerical_Features"][feature]['mutual_info'] = mi[counter]
                    counter = counter + 1
                    # Calculate missing values
                    self.json_data["Features"]["Numerical_Features"][feature]['missing_values'] = self.miss_value[
                        feature]
                    # Calculate min order and max order
                    self.json_data["Features"]["Numerical_Features"][feature]['min_orderm'] = self.min_orderm_cal(
                        self.csv_data, feature)
                    self.json_data["Features"]["Numerical_Features"][feature]['max_orderm'] = self.max_orderm_cal(
                        self.csv_data, feature)
                    # Calculate the correlation between selected feature and target feature.
                    #if 'numeric' in self.target_feature_type or 'Numeric' in self.target_feature_type:
                    #    self.json_data["Features"]["Numerical_Features"][feature]['correlation'] = self.corr_cal(self.csv_data, feature)
                    #elif 'categoric' in self.target_feature_type or 'Categoric' in self.target_feature_type or 'binary' in self.target_feature_type or 'Binary' in self.target_feature_type:
                    #    (pval, chisq_correlated) = self.chisq_correlated_cal(self.csv_data, feature)
                    #    self.json_data["Features"]["Numerical_Features"][feature]['Correlation'] = {}
                    #    self.json_data["Features"]["Numerical_Features"][feature]['Correlation']['chisq_correlated'] = chisq_correlated
                    #    self.json_data["Features"]["Numerical_Features"][feature]['Correlation']['p_val'] = pval
                    # Calculate IQR and Quartiles
                    q0, q1, q2, q3, q4, iqr = self.iqr_cal(self.csv_data, feature)
                    self.json_data["Features"]["Numerical_Features"][feature]['Quartiles'] = {}
                    self.json_data["Features"]["Numerical_Features"][feature]['Quartiles']['q0'] = q0
                    self.json_data["Features"]["Numerical_Features"][feature]['Quartiles']['q1'] = q1
                    self.json_data["Features"]["Numerical_Features"][feature]['Quartiles']['q2'] = q2
                    self.json_data["Features"]["Numerical_Features"][feature]['Quartiles']['q3'] = q3
                    self.json_data["Features"]["Numerical_Features"][feature]['Quartiles']['q4'] = q4
                    self.json_data["Features"]["Numerical_Features"][feature]['Quartiles']['iqr'] = iqr
                    # Calculate outlier info
                    outlier_list = self.detect_outlier(self.csv_data, feature)
                    self.json_data["Features"]["Numerical_Features"][feature]['Outliers'] = {}
                    self.json_data["Features"]["Numerical_Features"][feature]['Outliers']['number'] = len(outlier_list)
                    self.json_data["Features"]["Numerical_Features"][feature]['Outliers'][
                        'Actual_Values'] = outlier_list
                    # Distribution Check
                    self.json_data["Features"]["Numerical_Features"][feature]['Distribution'] = {}
                    normal_distrn_bool = self.shapiro_test_normality(self.csv_data, feature)
                    self.json_data["Features"]["Numerical_Features"][feature]['Distribution']['normal'] = normal_distrn_bool
                    self.json_data["Features"]["Numerical_Features"][feature]['Distribution']['exponential'] = self.ks_test_exponential(self.csv_data, feature)
                    if normal_distrn_bool:
                        self.json_data["Features"]["Numerical_Features"][feature]['Distribution']['skewness'] = stats.skew(self.csv_data[feature])
                else:
                    self.json_data["Info"]["discarded_features"].append(feature)
                    print(feature + " is dropped for having missing values more than 1/4 the whole size of the dataset")
            return ("analysis successfully completed")
        except TypeError:
            print("Numeric Feature Analysis Terminated")
            print("Please recheck feature type of feature: " + feature)
            return (feature)

    # Calculate parameters for categorical features and add it to json
    def analyse_categorical_fatures(self):
        print("Analysing categorical features")
        # Calculate parameters for categorical features and add it to json
        self.json_data["Features"]["Categorical_Features"] = {}
        categorical_features = pd.DataFrame()
        for i in range(len(self.categorical_features)):
            if self.csv_data_complete.columns[self.categorical_features[i]] in self.csv_data.columns:
                self.csv_data[self.csv_data_complete.columns[self.categorical_features[i]]]= self.csv_data[self.csv_data_complete.columns[self.categorical_features[i]]].astype('category')
                categorical_features = pd.concat([categorical_features, self.csv_data[self.csv_data_complete.columns[self.categorical_features[i]]].cat.codes],axis=1)
        if not (len(categorical_features) == 0):
            mi = mutual_info_classif(categorical_features, self.csv_data[self.class_label])
        counter = 0
        for column_nr in self.categorical_features:
            feature = self.column_names_list[column_nr]
            if feature not in self.drop_cols:
                self.json_data["Info"]["analyzed_features"].append(feature)
                self.json_data["Features"]["Categorical_Features"][feature] = {}
                # Calculate missing values
                self.json_data["Features"]["Categorical_Features"][feature]['missing_values'] = self.miss_value[feature]
                # Identify levels
                (index_list, val_list, num_levels) = self.freq_counts(self.csv_data, feature)
                levels = {}
                # Mongodb does not accept key name with dots.
                for i in range(len(val_list)):
                    if "." in str(index_list[i]):
                        index_list[i] = str(index_list[i]).replace(".", "")
                    levels[str(index_list[i])] = str(val_list[i])
                self.json_data["Features"]["Categorical_Features"][feature]['nr_levels'] = num_levels
                self.json_data["Features"]["Categorical_Features"][feature]['Levels'] = levels
                # Calculate imbalance
                imbalance = self.imbalance_test(self.csv_data, feature)
                self.json_data["Features"]["Categorical_Features"][feature]['imbalance'] = imbalance
                # Assign the mutual information for the feature
                self.json_data["Features"]["Categorical_Features"][feature]['mutual_info'] = mi[counter]
                counter = counter + 1
                # Calculate correlation between selected feature and target feature.
                #(pval, chisq_correlated) = self.chisq_correlated_cal(self.csv_data, feature)
                #self.json_data["Features"]["Categorical_Features"][feature]['Correlation'] = {}
                #self.json_data["Features"]["Categorical_Features"][feature]['Correlation']['p_val'] = pval
                #self.json_data["Features"]["Categorical_Features"][feature]['Correlation']['chisq_correlated'] = chisq_correlated
                # Implement the monotonous filtering
                self.json_data["Features"]["Categorical_Features"][feature]['monotonous_filtering'] = self.monotonous_filtering_categorical(self.csv_data, feature)
            else:
                self.json_data["Info"]["discarded_features"].append(feature)
                print(feature + " is dropped for having missing values more than 1/4 the whole size of the dataset")

    def analyse_text_features(self):
        print("Analysing text features")
        self.json_data["Features"]["Text_Features"] = {}
        #features = self.unstructured_features
        for column_nr in self.unstructured_features:
            feature = self.column_names_list[column_nr]
            if feature not in self.drop_cols:
                self.json_data["Info"]["analyzed_features"].append(feature)
                self.json_data["Features"]["Text_Features"][feature] = {}
                # Calculate missing values
                self.json_data["Features"]["Text_Features"][feature]['missing_values'] = self.miss_value[feature]
                (vocab_size, relative_vocab, vocab_concentration, entropy, min_vocab, max_vocab) = self.text_statistics(self.csv_data, feature)
                self.json_data["Features"]["Text_Features"][feature]["vocab_size"] = vocab_size
                self.json_data["Features"]["Text_Features"][feature]["relative_vocab"] = relative_vocab
                self.json_data["Features"]["Text_Features"][feature]["vocab_concentration"] = vocab_concentration
                self.json_data["Features"]["Text_Features"][feature]["entropy"] = entropy
                self.json_data["Features"]["Text_Features"][feature]["min_vocab"] = min_vocab
                self.json_data["Features"]["Text_Features"][feature]["max_vocab"] = max_vocab

            else:
                self.json_data["Info"]["discarded_features"].append(feature)
                print(feature + " is dropped for having missing values more than 1/4 the whole size of the dataset")

    def datetime_features_computations(self, df, col_name):
        feature = df[col_name]
        sorted_feature = feature.sort_values(ascending=True, ignore_index=True)
        difference_dates = sorted_feature.diff()
        min_value = difference_dates.min()
        max_value = difference_dates.max()
        median_value = difference_dates.median()
        mean_value = difference_dates.mean()


        for i, date in enumerate(sorted_feature):
            sorted_feature[i] = datetime.datetime.fromtimestamp(sorted_feature[i])

        daypart_frequencies = np.zeros(3)
        month_frequencies = np.zeros(12)
        weekday_frequencies = np.zeros(7)
        hour_frequencies = np.zeros(24)
        for i in sorted_feature:
            if i.time().hour > 3 and i.time().hour <= 12:
                if i.time().hour == 12:
                    if i.time().minute == 0:
                        daypart_frequencies[0] += 1
                else:
                    daypart_frequencies[0] += 1
            elif i.time().hour >= 12 and i.time().hour <= 20:
                if i.time().hour == 12:
                    if not i.time().minute == 0:
                        daypart_frequencies[1] += 1
                elif i.time().hour == 20:
                    if i.time().minute == 0:
                        daypart_frequencies[1] += 1
                else:
                    daypart_frequencies[1] += 1
            else:
                daypart_frequencies[2] += 1

            month_frequencies[(i.date().month)-1] +=1
            weekday_frequencies[i.weekday()] +=1
            hour_frequencies[(i.time().hour)-1] +=1

        return min_value, max_value, mean_value, median_value, daypart_frequencies, month_frequencies, weekday_frequencies, hour_frequencies

    def analyse_datetime_features(self):
        print("Analysing datetime features")
        self.json_data["Features"]["Datetime_Features"] = {}
        dayparts = ['daypart_morning','daypart_afternoon','daypart_evening']
        months = ['month_january','month_february','month_march','month_april','month_may','month_june','month_july','month_august','month_september','month_october','month_novmber','month_december']
        days = ['week_monday','week_tuesday','week_wednesday','week_thursday','week_friday','week_saturday','week_sunday']
        hours = ['hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6','hour_7','hour_8','hour_9','hour_10','hour_11','hour_12','hour_13','hour_14','hour_15','hour_16','hour_17','hour_18','hour_19','hour_20','hour_21','hour_22','hour_23',]
        for column_nr in self.datetime_features:
            feature = self.column_names_list[column_nr]
            if feature not in self.drop_cols:
                self.json_data["Info"]["analyzed_features"].append(feature)
                self.json_data["Features"]["Datetime_Features"][feature] = {}
                # Calculate missing values
                self.json_data["Features"]["Datetime_Features"][feature]['missing_values'] = self.miss_value[feature]
                min_value, max_value, mean_value, median_value, daypart_frequencies, month_frequencies, weekday_frequencies, hour_frequencies = self.datetime_features_computations(self.csv_data, feature)
                self.json_data["Features"]["Datetime_Features"][feature]['min_delta'] = min_value
                self.json_data["Features"]["Datetime_Features"][feature]['max_delta'] = max_value
                self.json_data["Features"]["Datetime_Features"][feature]['mean_delta'] = mean_value
                self.json_data["Features"]["Datetime_Features"][feature]['median_delta'] = median_value
                for i, value in enumerate(daypart_frequencies):
                    self.json_data["Features"]["Datetime_Features"][feature][
                        dayparts[i]] = value
                for i,value in enumerate(month_frequencies):
                    self.json_data["Features"]["Datetime_Features"][feature][
                        months[i]] = value
                for i,value in enumerate(weekday_frequencies):
                    self.json_data["Features"]["Datetime_Features"][feature][
                        days[i]] = value
                for i,value in enumerate(hour_frequencies):
                    self.json_data["Features"]["Datetime_Features"][feature][
                        hours[i]] = value

            else:
                self.json_data["Info"]["discarded_features"].append(feature)
                print(feature + " is dropped for having missing values more than 1/4 the whole size of the dataset")




    # Write result to MongoDB
    def write_result_to_DB(self):
        # Connect to Database and write json data
        db_write_status = ''
        print("Connected to database")
        myclient = pymongo.MongoClient("mongodb://admin:admin@localhost:27017/")
        try:
            # List all databases
            db_list = myclient.list_database_names()
            print("Connection successful. Databases available:")
            print(db_list)
        except pymongo.errors.ConnectionError as e:
            print("Connection failed:", e)

        dbname = myclient["assistml"]
        self.collection_datasets = dbname["datasets"]
        data_already_in_db = self.check_data_availability_in_db(json.loads(self.json_data))
        if data_already_in_db:
            print("Information about this dataset is already available in DB. Skipping insertion.")
            db_write_status = 'Information about this dataset is already available in Database. Insertion skipped'
        else:
            mylist = []
            mylist.append(json.loads(self.json_data))
            self.collection_datasets.insert_many(mylist)
            db_write_status = 'Data written succcessfully to Database'
            print("Data written succcessfully to MongoDB")
        return self.json_data, db_write_status

    # Main function which invokes all the other functions
    def analyse_dataset(self):
        print("Analysing Dataset")
        start = time.time()
        processing_status = self.process_pandas_df()
        if not "processing success" in processing_status:
            error_message = "Please recheck target class label"
            return {}, error_message
        parse_feature_status = self.process_feature_annotation_list()
        if not "parsing success" in parse_feature_status:
            print("Parsing Failed")
            error_message = "Please recheck feature type of the feature: " + parse_feature_status
            return {}, error_message
        self.calculate_ratios()
        analysis_status = self.analyse_numerical_features()
        if not "analysis success" in analysis_status:
            print("Analysis Failed")
            error_message = "Please recheck feature type of the feature: " + analysis_status
            return {}, error_message
        self.analyse_categorical_fatures()
        self.analyse_text_features()
        self.analyse_datetime_features()
        stop = time.time()
        analysis_time = stop - start
        print(analysis_time)
        self.json_data["Info"]["analysis_time"] = analysis_time
        #print(self.json_data["Info"]["analysis_time"])
        self.json_data = json.dumps(self.json_data, indent=4, default=self.datatype_converter)
        json_output, db_write_status = self.write_result_to_DB()
        return json_output, db_write_status

    # Write json data to a file in local directory
    '''with open(filename, 'w') as f:
        json_data = json.dumps(json_data,indent=4, default=datatype_converter)
        print(json_data)
        f.write(json_data)'''


if __name__ == '__main__':
    # Command line arguments
    mode = int(sys.argv[1])
    df = ''
    dataset_path = ''
    if mode == 1:
        dataset_path = sys.argv[2]
    elif mode == 2:
        df = sys.argv[2]
    else:
        print("Invalid argument for Mode. Please try again with valid input")
        print("Accepted values : 1 or 2")
        exit(1)
    dataset_name = sys.argv[3]
    # print(dataset_name)
    target_label = sys.argv[4]
    # print(target_label)
    target_feature_type = sys.argv[5]
    # print(target_feature_type)
    use_case = sys.argv[6]
    # print(use_case)
    feature_annotation_list = sys.argv[7]
    # print(feature_annotation_list)
    data_profiler = DataProfiler(mode, dataset_path, df, dataset_name, target_label, target_feature_type, use_case,
                                 feature_annotation_list)
    data_profiler.analyse_dataset()
