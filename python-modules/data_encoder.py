
import sys
import pandas as pd
import numpy as np
import pymongo


def process_cmd_arg(cmd_argument, available_elements_db):
    annotation_list = cmd_argument.replace(' ', '')
    annotation_list = annotation_list.replace("'", '')
    annotation_list = annotation_list.replace('"', '')
    annotation_list = list(filter(None,annotation_list.strip('[]').split(',')))
    if(len(annotation_list) != 0):
        unrecognised_elements = []
        for element in annotation_list:
            if element not in available_elements_db:
                unrecognised_elements.append(element)
        for element in unrecognised_elements:
            annotation_list.remove(element)
    return annotation_list


def main():

    # Database connection
    myclient = myclient = pymongo.MongoClient("mongodb://admin:admin@localhost:27017/")
    dbname = myclient["assistml"]
    collection_enriched = dbname["enriched_models"]
    print("Connected to database")
    data = pd.DataFrame(list(collection_enriched.find({},{"_id":0})))
    column_names_enriched = data.columns
    model_names_enriched = collection_enriched.distinct("model_name")

    data_new = pd.DataFrame([])
    # Select model names (Row names) for analysis
    if len(sys.argv) > 2:
        model_names_list = sys.argv[2]
        models_list = process_cmd_arg(model_names_list, model_names_enriched)
        if len(models_list) != 0:
            print("Found model names list in cmd argument")
            for model_name in models_list:
                # Create a subset DataFrame for the model_name
                subset = data[data.model_name == model_name]
                # Concatenate subset to data_new
                data_new = pd.concat([data_new, subset], ignore_index=True)
            print(data_new)
        else:
            print("Using all models for csv generation")


    # Select column names for analysis
    new_file_name=""
    if len(sys.argv) >1:
        feature_annotation_list = sys.argv[1]
        column_list = process_cmd_arg(feature_annotation_list, column_names_enriched)
        if(len(column_list) != 0):
            print("Found feature annotation list in cmd argument")
            data_new = data_new.loc[:,column_list]
        else:
            print("Using default feature annotation list")
            data_new = data_new.loc[:,["fam_name","rows","sampling","language","nr_hyperparams_label","performance_gap","test_size","categorical_encoding"]]
    else:
        print("Using default feature annotation list")
    print(data_new)


    output_file_name="merged_data_" + new_file_name +"selectedcols.csv"
    data_new.to_csv(output_file_name, index=False)



if __name__ == "__main__":
    main()
