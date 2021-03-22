# Music Genre Classification
## Introduction
This project was completed during my time on the General Assembly Data Science Immersive course.

Through my analysis of the Free Music Archive audio files and use of the Python package, Librosa. I aim to determine the class of music genre of an audio file. As well as understand the key predictors of this type of classification problem.

I performed various statistical and machine learning models for the classification problem including one hyperparameter grid search on AWS. The model that generalises best predicted the music genre with a 65% accuracy compared to the baseline of 30%. Key insights and limitations in accuracy were found, such as increase of bias from class imbalance, inaccurate tagging of metadata for model training, limit of explained variance due to required dimensionality reduction.

Future improvements and general limitation are summarised below.
## Files in this Repository
Presentation slides: Used to present the project, findings, limitations and future improvements to a non-technical audience.
Technical report: Report aimed for a technical audience. It contains a detailed explanation of the extraction methods employed and general methodology, exploratory data analysis, preprocessing steps, modelling stage, findings,  limitations and improvements for the future.


 - [Presentation Slides](http://www.google.fr/ "Presentation Slides") : Used to present the project, findings, limitations and future improvements to a non-technical audience.
 
 - [Technical Report](http://www.google.fr/ "Technical Report") : Report aimed for a technical audience. It contains a detailed explanation of the extraction methods employed and general methodology, exploratory data analysis, preprocessing steps, modelling stage, findings,  limitations and improvements for the future.
 - Jupyter Notebook files (.ipynb)
   - 1.0 : [Dataset Inspection]()  (dataset_meta_data_inspection.ipynb)
   - 2.0 : Features
    - [Feature inspection & Librosa]() (feature_inspection_librosa.ipynb)
    - [Feature extraction]() (ATS_feature_extraction.ipynb)
   - 3.0 : Features
    - Methodology
      - [SS]() (method_1_SS.ipynb)
      - [PCA]() (method_2_PCA.ipynb)
      - [Collating genres]() (folders 51 - 100) collate_genre_meta_into_df_51_100.ipynb
   - 4.0 : Features
      - [SS]() (EDA_1_SS.ipynb)
      - [PCA]() (EDA_2_PCA.ipynb)
   - 5.0 : Modelling
      - [SS]() (modelling_1_SS.ipynb)
      - [PCA]() (modelling_2_PCA.ipynb)
   - 6.0 : [Evaluation (SS)]() (evaluation_1_SS.ipynb)
   - 7.0 : [Further EDA (SS)]() (further_EDA_1_SS.ipynb)
   - 8.0 : Unsupervised learning
      - [SS]() (unsupervised_learning_method_1_SS.ipynb)
      - [PCA]() (unsupervised_learning_method_2_PCA.ipynb)
## Problem statement
Put very simply the main question I am asking is “ Is there a way to automate music genre classification?”. Ultimately having to manually assign genre is still currently how labels are assigned and is crucial to automate metadata in large music databases as the music industry accelerates into the digital realm.

## Main Objectives
My main objectives for this project were to: 
  1. Build a 10+ music genre classifier with > 70% accuracy
  2. Compare unsupervised classifier clusters to actual
  3. Classify sub-genres or styles

## Dataset
FMA Dataset
https://github.com/mdeff/fma/blob/f6a8faf7cbfffa967f40c37cf1fddba908696514/README.md

I used publicly available data on [FMA](https://github.com/mdeff/fma). I downloaded the FMA medium zip file which includes:
  - 16,000 tracks (22 GiB), mp3 format, 30s length
  - 16 genres Unbalanced Genres

The FMA medium dataset is a very unbalanced dataset with Rock and Blues having a count of 4000+ and 200 respectively. A few genres also did not seem to represent a modern genre and decided not to include them ;  ‘Easy listening’, ‘Spoken’,  ‘Old-Time / Historic’ and ‘International’. 

Ended up reducing to ~14,800 tracks, covering 12 genres (12 class problem)


**Model  | Scores**
------------- | -------------
Random Forest  | 0.64
SVM  | 0.61
KNN | 0.58

