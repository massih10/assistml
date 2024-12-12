
# Needed when not deployed as package and still launched from RStudio
source("cluster.R")
source("distance.R")
source("query.R")
source("rank.R")
source("results.R")
source("select.R")
source("reticulate.R")
source("explain.R")

#* Enables data upload of CSV files
#* @post /upload
upload_data <- function(req) {
  # Parse the request to extract file information
  files <- Rook::Multipart$parse(req)
  
  # Debugging: Print the structure of the parsed files
  print("Structure of uploaded file:")
  print(str(files))
  
  # Access the uploaded file's temporary path
  uploaded_file <- files[[1]]$tempfile
  
  # Check if the file exists
  if (file.exists(uploaded_file)) {
    # Read the CSV file
    newcsv <- read.csv(uploaded_file, header = TRUE, as.is = TRUE, dec = ".")
    
    # Display a sample of the uploaded data
    print("Sample of uploaded data:")
    print(head(newcsv))
    
    # Save the uploaded data locally
    output_file <- files[[1]]$filename
    write.csv(x = newcsv, file = output_file, row.names = FALSE)
    
    print(paste("Just uploaded and saved as:", output_file))
    return(paste("File uploaded and saved as:", output_file))
  } else {
    stop("Uploaded file not found.")
  }
}

#* @post /assistml
#' @param classif_type String to say if it is binary or multiclass
#' @param classif_output String to say if the result should be a single prediction or class probabilities
#' @param dataset CSV file of the new use case with sample data to analyze
#' @param sem_types JSON(?) Array with annotations for the semantic types of the data features of dataset
#' @param accuracy_range Float to state the range of top models in the accuracy dimension to be considered acceptable (0 to 1)
#' @param precision_range Float to state the range of top models in the precision dimension to be considered acceptable (0 to 1)
#' @param recall_range Float to state the range of top models in the recall dimension to be considered acceptable (0 to 1)
#' @param trtime_range Float to state the range of top models in the training time dimension to be considered acceptable (0 to 1)
#'
#' @title AssistML analysis for new usecase/data
#' @description Recommends ML models for a given query based on a base of known trained model. Stitches up all functions from the package.
#'
#' @return Ranked list \code{placeholder} of relevant models as well as insights to consider when using them.
#' @export
#'
#' @examples
assistml<-function(classif_type, # For query usecase
                   classif_output,
                   sem_types,
                   accuracy_range, # For query preferences
                   precision_range,
                   recall_range,
                   trtime_range,
                   dataset_name,
                   usecase, # For query data
                   ...,
                   deployment, # For query data. Optional at the moment
                   lang, # For query settings. All optional.
                   algofam,
                   platform,
                   tuning_limit,
                   implementation # For query usecase. Optional
                   ){

  # assistml_log<-file("assistml_log.log") # File name of output log
  # sink(assistml_log, append = TRUE, type = "output") # Writing console output to log file
  # sink(assistml_log, append = TRUE, type = "message")

verbose=T
start.time <- Sys.time()
  print("Connecting to mongo to get base models")
  base_models<-mongolite::mongo(collection = "base_models",
                                db = "assistml",
                                url = "mongodb://admin:admin@localhost:27017/")

  # Obtaining default values to complete the query
  defaults<-base_models$find('{}',
                             fields='{"Model.Training_Characteristics.Dependencies.Platforms":1,
                             "Model.Training_Characteristics.Dependencies.Libraries":1,
                 "Model.Training_Characteristics.Hyper_Parameters.nr_hyperparams":1,
                             "Model.Training_Characteristics.implementation":1}')


  print("Connecting to mongo to get enriched models")
  enriched_models<-mongolite::mongo(collection = "enriched_models",
                                    db="assistml",
                                    url="mongodb://admin:admin@localhost:27017/")


  # more_defaults<-read.csv("quantile_binary.csv",header = T)
  more_defaults<-enriched_models$find(query = '{}') #Picking data from Mongo

  print(" ")
  # print(dataset)
  print(" ")


  # Starting connection to the queries database to store this one.
  querydb<-mongolite::mongo(db="assistml",collection = "queries",url="mongodb://admin:admin@localhost:27017/")
  queryId<-querydb$count()+1

  # Inserting query record before doing analysis
  print("Forming query record with fields...")
  print(paste("Query NR",queryId,"Issued at",as.character(Sys.time()),"For classification",classif_type))
  # print(classif_output)
  # print("")
  query_record<-list(
    "number"=queryId,
    "madeat"=paste0(gsub("-","",unlist(strsplit(as.character(Sys.time())," "))[1]),"-",substr(gsub(":","",unlist(strsplit(as.character(Sys.time())," "))[2]),1,4) ), #Gets timestamp in format 20211231-1457
    "classif_type"=classif_type,
    "classif_output"=classif_output,
    "dataset"=dataset_name, # For query data
    "semantic_types"=sem_types,
    "accuracy_range"=accuracy_range, # For query preferences
    "precision_range"=precision_range,
    "recall_range"=recall_range,
    "traintime_range"=trtime_range
  )

  # Data structures to compute the overal distrust score
  warnings<-list()
  distrust_pts<-0


  querydb$insert(data =
                   rjson::toJSON(query_record))

  print("#### QUERY FUNCTIONS ####")

  if(F){ # Change for verbose
    print("API call values")
    print(classif_type)
    print(classif_output)
    print(" ")

  }


  usecase_info<-query_usecase(classif_type,classif_output)

  if(F){ # Change to verbose
    print(paste(timestamp(prefix = "",suffix = ""),": Created use case info"))
    print(usecase_info)
    print(paste(" "))
  }

  data_feats<-query_data(dataset = newdata,semantic_types = sem_types,dataset_name=dataset_name,use_case=usecase)

  if(verbose){
    writeLines(paste("Retrieved descriptive data features for new dataset : \ncols",data_feats$features,"and \nrows",data_feats$observations))
  }

  print(" ")

  usecase_preferences<-query_preferences(accuracy_range, precision_range, recall_range, trtime_range)
  if(F){ # Change to verbose
    print("Created performance preferences list:")
    print(usecase_preferences)
    print(" ")
  }


  ## Call select.R functions   ####
  print(" ")
  print("#### SELECT FUNCTIONS ####")

  usecase_models<-choose_models(task_type = usecase_info$tasktype,
                output_type =  usecase_info$output,
                data_features = data_feats)

  if(is.null(usecase_models[[2]]) ){
    print("No similar models could be found")
    return("No similar models could be found")
  }else{
    distrust_pts<-distrust_pts+ (3-usecase_models[[2]])

    warnings[[length(warnings)+1]]<-switch (usecase_models[[2]]+1,
                                            "Dataset similarity level 0. Only the type of task and output match. Distrust Pts increased by 3",
                                            "Dataset similarity level 1. Datasets used shared data types. Distrust Pts increased by 2",
                                            "Dataset similarity level 2. Datasets used have similar ratios of data types. Distrust Pts increased by 1",
                                            "Dataset similarity level 3. Datasets used have features with similar meta feature values. Distrust Pts increased by 0"
    )
  }



  print(paste("assist(): Selected models:",nrow(usecase_models[[1]]),"found with similarity level",usecase_models[[2]]," ."))


  ## Call cluster.R function ####
  print(" ")
  print("#### CLUSTER FUNCTIONS ####")

  usecase_mgroups<-cluster_models(selected_models = usecase_models[[1]],preferences = usecase_preferences)
  print(paste("Acc Models:",length(usecase_mgroups$acceptable_models) ))

  # Checking if there are any NACC models
  if(usecase_mgroups$nearly_acceptable_models[1] %in% c("none")){
    print( paste("Nearly Acc Models:",usecase_mgroups$nearly_acceptable_models[1] ))
  }else{
    print( paste("Nearly Acc Models:",length(usecase_mgroups$nearly_acceptable_models) ))
  }


  # Adding warnings based on the cutoff of the clusters in each ACC and NACC region
  if(usecase_mgroups$distrust$acceptable_models>0){
    distrust_pts<-distrust_pts+usecase_mgroups$distrust$acceptable_models
    print(paste("Added",usecase_mgroups$distrust$acceptable_models," distrust points from ACC cut"))
    warnings[[length(warnings)+1]]<-paste0("The selection of ACC solutions was not as clean as possible. Distrust Pts increased by ",usecase_mgroups$distrust$acceptable_models)
  }

  if(usecase_mgroups$distrust$nearly_acceptable_models>0){
    distrust_pts<-distrust_pts+usecase_mgroups$distrust$nearly_acceptable_models
    print(paste("Added",usecase_mgroups$distrust$nearly_acceptable_models," distrust points from NACC cut"))
    if(verbose==T){print(paste0("The selection of NACC solutions was not as clean as possible. Distrust Pts+",usecase_mgroups$distrust$nearly_acceptable_models))}
    warnings[[length(warnings)+1]]<-paste0("The selection of NACC solutions was not as clean as possible. Distrust Pts increased by ",usecase_mgroups$distrust$nearly_acceptable_models)
  }


  ## Call rules.R functions####
  print(" ")

  print("#### RULES FUNCTIONS #### \n\n\n")


  if(usecase_mgroups$nearly_acceptable_models[1] %in% c("none")){
    usecase_rules<-python_rules(usecase_mgroups$acceptable_models)
  }else{
    usecase_rules<-python_rules(c(usecase_mgroups$acceptable_models,usecase_mgroups$nearly_acceptable_models))
  }


  print(paste("Finished rules generation.",length(usecase_rules),"rules generated."))

  ## Call rank.R functions####
  print(" ")

  print("#### RANK FUNCTIONS ####")

  ranked_models<-rank_models(usecase_mgroups,usecase_preferences)

  models_choice<-shortlist_models(ranked_models)


  ## Call results.R functions####
  print(" ")

  print("#### RESULTS FUNCTIONS ####")


  distrust_basis<-3+3+3 # 3 points for dataset similarity, 3 for ACC cut and 3 for NACC cut. ALWAYS
  print("Distrust score calculation")
  print(paste(distrust_pts,"over",distrust_basis))


  models_report<-generate_results(models_choice,usecase_rules,warnings,distrust_pts,query_record,distrust_basis)


  write(rjson::toJSON(models_report,indent = 3),"models_report.json")

  querydb$update(
    query=eval(parse(text = paste0("'{\"number\":",queryId,"}'"))),
                 update= eval(parse(text = paste0(
                   "'{\"$set\":{\"report\":",rjson::toJSON(models_report),"}}'" ))),
                 upsert=F,
                 multiple=F)

  print("ASSISTML TERMINATED")
  # close.connection(assistml_log)

  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print(paste("Time taken for end to end execution",time.taken))

  return(rjson::toJSON(models_report))


}

#* Echo the parameter that was sent in
#* @param msg The message to echo back.
#* @get /test
function(msg){
  if(missing(msg)) msg="pal"
  # assist_test<-file("assist_test.log")
  # sink(assist_test, append = TRUE, type = "output") # Writing console output to log file
  # sink(assist_test, append = TRUE, type = "message")


  print(paste0(timestamp(),": AssistML working, ",msg))
  print("")
  print(sessionInfo())
  print("")
  print(osVersion)

  # close.connection(assist_test)

  return(paste0(timestamp(),": AssistML working, ",msg))
}
