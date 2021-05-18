# AssistML: A concept to recommend ML solutions for predictive use cases

> 18.05.2021

<img src="./assistML.png"/>



## Abstract

The adoption of machine learning (ML) in organizations is characterized by the use of multiple ML software components. Citizen data scientists face practical requirements when building ML systems, which go beyond the known challenges of ML, e. g., data engineering or parameter optimization. They are expected to quickly identify ML system options that strike a suitable trade-off across multiple performance criteria. These options also need to be understandable for non-technical users. Addressing these practical requirements represents a problem for citizen data scientists with limited ML experience. This calls for a method to help them identify suitable ML software combinations. Related work e. g., AutoML systems, are however not responsive enough or cannot balance different performance criteria. In this paper, we introduce AssistML, a novel concept to recommend ML solutions, i. e., software systems with ML models, for predictive use cases. AssistML uses metadata of existing ML solutions to quickly identify and explain options for a new use case. We implement the approach and evaluate it with two exemplary use cases. Results show that AssistML provides recommendations with performance in line with users’ preferences.

## Code organization

- **python-modules**: Main script is assist-dashboard.py
- **r-api**: R modules are located in this directory. Main script is assist.R
- **repository**: Contains metadata to recreate the metadata repository
- **source-code**: Contains the corresponding source code of the ML solutions described in the metadata repository.





## Further notes

A demonstration video is provided: *[q-steel-20.mkv](q-steel-20.mkv)*

Currently under submission. Arxiv will be made available upon acceptance.