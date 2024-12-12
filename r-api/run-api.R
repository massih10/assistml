library(plumber)

# Initialize the API
r <- plumb("assist.R")  # Reference the main script
r$run(port = 4321)
