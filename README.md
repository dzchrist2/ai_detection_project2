# AI-Generated Text Classification for Scientific Articles

### Install
This project requires **Python** and the following Python libraries to be installed:
 - [Pandas](http://pandas.pydata.org/)
 - [scikit-learn](http://scikit-learn.org/stable/)

### Code

There are 2 Python code files for this project. The first is `experimentation.py` which is the file used to explore and experiment with various feature extraction methods and classification algorithms. This file requires a CVS dataset named `train.csv` which will be the data from which the features are extracted. The second is `ai_detection.py` which is the file containing the complete machine model which classifies scientific text as Human(0) or AI-generated(1). This file requires two CSV datasets, one to train on, named `train.csv`, and one to perform predictions on, named `test.csv`. This program will generate a CSV file which contains the predictions made by the model, named `results.csv`. 

### Run

In a terminal or command window, navigate to the top level of the project directory `ai_detection_project2`. To run `experimentation.py` execute the following command: 

```bash
python experimentation.py
```

To run `ai_detection.py` execute the following command: 

```bash
python ai_detection.py
```

### Data

The 2 necessary datasets for this program, `train.csv` and `test.csv`, are saved within a directory named `data`. These datasets are stored in CSV format and have the following columns:
 - ID: a number representing the scientific article
 - Title: Contains the text from the title of the article
 - Abstract: Contains the text from the abstract section of the article
 - Introduction: Contains the text from the introduction section of the article
 - Label: (train.csv only) Contains labels of 0 for human generated or 1 for AI generated 

The CSV file generated by `ai_detection.py` is saved within the top level of the project directory. This file has the following 2 columns:
 - ID: a number representing the scientific article
 - Label: Contains labels generated by the model, 0 for human generated or 1 for AI generated 
