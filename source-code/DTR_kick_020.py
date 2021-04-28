{
	"DTR_kick_020": {
		"Data_Meta_Data": {
			"Name": "kick",
			"Rows": 69803,
			"Classification Type": "Binary",
			"Class Variable": "IsBadBuy",
			"Categorical Columns": 18,
			"Numeric Columns": 13,
			"Columns": 33,
			"Datetime Columns": 1,
			"Text Columns": 0,
			"Preprocessing": {
				"categorical_encoding": "One-Hot Encoding",
				"numerical_encoding": "as-is",
				"numeric_imputation": "median",
				"numeric_selection": "Numerical Analysis Metrics",
				"categorical_selection": "Categorical Analysis Metrics",
				"Numeric_properties": {
					"VehYear": {
						"number_outliers": 0,
						"missing_values": 0,
						"correlation": -0.16176565245184024
					},
					"MMRAcquisitionAuctionAveragePrice": {
						"number_outliers": 261,
						"missing_values": 18,
						"correlation": -0.090179179775925
					},
					"MMRAcquisitionAuctionCleanPrice": {
						"number_outliers": 1543,
						"missing_values": 18,
						"correlation": -0.1052186074575074
					},
					"MMRAcquisitionRetailAveragePrice": {
						"number_outliers": 261,
						"missing_values": 18,
						"correlation": -0.090179179775925
					},
					"MMRAcquisitonRetailCleanPrice": {
						"number_outliers": 1132,
						"missing_values": 18,
						"correlation": -0.08605720306769234
					},
					"MMRCurrentAuctionAveragePrice": {
						"number_outliers": 502,
						"missing_values": 315,
						"correlation": -0.11193245979324842
					},
					"MMRCurrentAuctionCleanPrice": {
						"number_outliers": 1123,
						"missing_values": 315,
						"correlation": -0.10546742322090732
					},
					"MMRCurrentRetailAveragePrice": {
						"number_outliers": 233,
						"missing_values": 315,
						"correlation": -0.10156175780718084
					},
					"MMRCurrentRetailCleanPrice": {
						"number_outliers": 829,
						"missing_values": 315,
						"correlation": -0.09721550440572851
					},
					"VehBCost": {
					    "number_outliers": 169,
						"missing_values": 68,
						"correlation": -0.10348681859882947
					}

				}
			},
			"Columns After Preprocessing": 1132,
			"List of Columns Used": "['IsBadBuy', 'PurchDate', 'VehicleAge', 'VehOdo','MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice','MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice','MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',...,'Size_MEDIUM SUV', 'Size_SMALL SUV', 'Size_SMALL TRUCK','Size_SPECIALTY', 'Size_SPORTS', 'Size_VAN','TopThreeAmericanName_CHRYSLER', 'TopThreeAmericanName_FORD','TopThreeAmericanName_GM', 'TopThreeAmericanName_OTHER']",
	    	"List of Columns dropped":"['AUCGUART','PRIMEUNIT','Trim','VNZIP1','VNST','BYRNO','SubModel','WheelTypeID','VehYear','Color']"

		},
		"Training Characteristics": {
			"Hyper Parameters": {
              "criterion": "gini",
                "splitter": "best",
                "n_estimators": "10",
                "Max_depth": "None",
                "Min_samples_split": "2"
			},
			"Test_size": 0.25,
			"No. of Cross Validation Folds Used": 5,
			"Sampling": "Upsampling",
			"Algorithm_Implementation": "sklearn.tree.RandomForestClassifier",
			"Language": "python",
			"Language Version": "3.6",
			"Dependencies": {
				"Platforms": {
					"conda": "4.5.12",
					"spyder": "4.1.1",
					"spyder-kernels": "1.9.0"
				},
				"Libraries": {
					"pandas": "1.0.3",
					"numpy": "1.18.2",
					"scikit-learn": "0.22.2.post1",
					"joblib": "0.14.1"
				}
			},
			"Cores": 2,
			"GhZ": 2.6,
			"deployment": "single",
			"Deployment": "single_host",
			"Implementation": "single_language"
		},
		"Metrics": {
			"Accuracy": 0.5,
			"Error": 0.5,
			"Precision": 0.5,
			"Recall": 0.5,
			"FScore":0.66666,
			"Single_Training_Time": 1.5719194412231445,
			"Cross-validated Training Time":1.3685378147125244,
			"Test time per unit": 2.5302294216014877e-06,
			"Confusion_Matrix" : [[15766,0],[15677,0]],
			"Test_File":"kick_upsampled_onehot_test_25.csv"
		}
	}
}
