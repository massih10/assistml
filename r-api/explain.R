explain_python<-function(model_code){
 verbose<-T

 if(verbose){
   print(paste("Generating explanation for",model_code))
 }


 script<-"import os\n"
 print("osversion")
 print(osVersion)

#  # Appending right path depending on OS
#  if(stringr::str_match(osVersion,"Windows") %in% "Windows"){
#    # For Windows local
#    script<-paste0(script,"os.system(\'python C:/Users/alexv/Documents/PhD/1prod/guacamole/asm-2/2_code/explainability.py -m ",model_code,"\')")
#    print(osVersion)
#  }else if(stringr::str_match(osVersion,"Ubuntu") %in% "Ubuntu"){
#    # For Ubuntu remote
#    script<-paste0(script,"os.system(\'python3.8 /home/ubuntu/asm-2/2_code/explainability .py -m ",model_code,"\')")
#    print(osVersion)
#  }
  script<-paste0(script,"os.system(\'/home/shoseini/assistml/venv-assistml/bin/python3.9 /home/shoseini/assistml/python-modules/explainability .py -m ",model_code,"\')")



 if(verbose){
    print(paste("Calling python script to generate explanations for",model_code))
 }

 write(script,file = "explain_gen.py",append = F)
 reticulate::py_run_file("explain_gen.py")

 if(verbose){
    print(paste("Retrieving explanations from Mongo for",model_code))
 }
 base <- mongolite::mongo(
    collection = "base_models",
    db = "assistml",
    url = "mongodb://admin:admin@localhost:27017/"
)
 return(
    base$find( query =eval(parse(text = paste0("'{\"Model.Info.name\":{\"$in\":[\"",model_code,"\"]}}'") )) )
 )

}
