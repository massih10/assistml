
python_rules<-function(model_codes){
  verbose<-T

  if(verbose){print("Entered python_rules() with the following chosen models")}
  if(missing(model_codes)) model_codes=""

  if(verbose){print(model_codes)
  print("")}

  # install.packages("reticulate")

  # reticulate::virtualenv_list()
  # reticulate::virtualenv_create("assistml")
  # reticulate::py_install("pymongo",envname = "assistml")
  # reticulate::py_install("mlxtend",envname = "assistml")
  # reticulate::use_virtualenv("assistml")
  # reticulate::use_virtualenv("guacamole")
  # reticulate::py_install("pymongo",envname = "guacamole")
  # reticulate::py_install("mlxtend",envname = "guacamole")

  # reticulate::py_install("pandas")
  # reticulate::py_install("pymongo")



  # Sys.which("python")

  # py_config()

  print("Calling python data_encoder.py...")
  # Triggers execution of python modules with mlxtend to generate rules

  data_enc<-"import os\n"

# Appending right path depending on OS
  if(stringr::str_match(osVersion,"Windows") %in% "Windows"){
    # For Windows local
    data_enc<-paste0(data_enc,"os.system(\'python C:/Users/alexv/Documents/PhD/2stdi/ws19-studyproject/mlm-perf/modules/data_encoder.py [\"fam_name\",\"nr_hyperparams_label\",\"performance_gap\",\"quantile_accuracy\",\"quantile_recall\",\"quantile_precision\",\"platform\",\"quantile_training_time\",\"nr_dependencies\"] [\"",paste0(model_codes,collapse = "\",\""),"\"]\')")
    print(osVersion)
  }else if(stringr::str_match(osVersion,"Ubuntu") %in% "Ubuntu"){
    # For Ubuntu remote
    data_enc<-paste0(data_enc,"os.system(\'python3.8 /home/ubuntu/mlm-perf/modules/data_encoder.py \"[fam_name,nr_hyperparams_label,performance_gap,quantile_accuracy,quantile_recall,quantile_precision,platform,quantile_training_time,nr_dependencies]\" \"[",paste0(model_codes,collapse = ","),"]\"\')")
    print(osVersion)
  }

  # paste0(c("SVM_kick_003","DTR_kick_012","RFR_kick_022","DTR_kick_011","NBY_bank_002"),collapse = "\",\"")

  write(data_enc,file = "data_rules_gen.py",append = F)
  reticulate::py_run_file("data_rules_gen.py")


  # print("Generating rules ...")
  rules_py<-"import os\n"


  # Ranking metric :: 0=confidence | 1=lift | 2=leverage | 3=Conviction
  # ranking metric, metric min score, min support
  if(stringr::str_match(osVersion,"Windows") %in% "Windows"){
    # For Windows local
    rules_py<-paste0(rules_py,"os.system(\'python C:/Users/alexv/Documents/PhD/2stdi/ws19-studyproject/mlm-perf/modules/association_python.py 0 0.7 0.25 \')")
    print(osVersion)
  }else if(stringr::str_match(osVersion,"Ubuntu") %in% "Ubuntu"){
    # For Ubuntu remote
    rules_py<-paste0(rules_py,"os.system(\'python3.8 /home/ubuntu/mlm-perf/modules/association_python.py 0 0.7 0.25 \')")
    print(osVersion)
  }



  write(rules_py,file = "rules.py",append = F)
  print("Calling python association_python.py")
  reticulate::py_run_file("rules.py")


  analysis_py<-"import os\n"


  if(stringr::str_match(osVersion,"Windows") %in% "Windows"){
    # For Windows local
    analysis_py<-paste0(analysis_py,"os.system(\'python C:/Users/alexv/Documents/PhD/2stdi/ws19-studyproject/mlm-perf/modules/analysis.py 0.5 0.01 1.2 \')")
    print(osVersion)
  }else if(stringr::str_match(osVersion,"Ubuntu") %in% "Ubuntu"){
    # For Ubuntu remote
    analysis_py<-paste0(analysis_py,"os.system(\'python3.8 /home/ubuntu/mlm-perf/modules/analysis.py 0.5 0.01 1.2 \')")
    print(osVersion)
  }


  write(analysis_py,file = "push_rules.py",append = F)
  print("Calling python analysis.py")
  reticulate::py_run_file("push_rules.py")

  print("Retrieving last added rules summary from Mongo")

  rulestamp<-unlist(strsplit(as.character(lubridate::now())," "))
  rulestamp<-paste0(gsub("-","",rulestamp[1]),"-",substr(gsub(":","",rulestamp[2]),1,4))
  
  if(verbose){
    print("Retrieving rules for experiment inserted at:")
    print(rulestamp)
  }
  



  rules<-mongolite::mongo("rules","assistml","mongodb://localhost")
  current_setofrules<-rules$find(query = paste0('{ "Rules":{"$exists":true}, "Experiment.created":"',rulestamp,'"}'),
                              fields = '{"Rules":true}'
                              )$Rules


  if(length(current_setofrules)>0){
    print("Storing rules as json")
    # Saves the found rules as Json for backup
    write(rjson::toJSON(current_setofrules[1:length(current_setofrules)],indent = 3),file = "python_rules.json",append = F)
    return(current_setofrules[1:length(current_setofrules)])
  }else{
    print("No rules were found nor filtered")
    return(current_setofrules)
  }




  }


