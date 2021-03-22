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
   - 3.0 : Methodology
      - [SS]() (method_1_SS.ipynb)
      - [PCA]() (method_2_PCA.ipynb)
      - [Collating genres]() (folders 51 - 100) collate_genre_meta_into_df_51_100.ipynb
   - 4.0 : EDA
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
I used the publicly available data from [FMA](https://github.com/mdeff/fma) originally found on [ISMIR](http://ismir.net/resources/datasets/). I downloaded the FMA medium zip file which includes:
  - 16,000 tracks (22 GiB), mp3 format, 30s length
  - 16 genres Unbalanced Genres

The FMA medium dataset is a very unbalanced dataset with Rock and Blues having a count of 4000+ and 200 respectively. A few genres also did not seem to represent a modern genre and decided not to include them ;  ‘Easy listening’, ‘Spoken’,  ‘Old-Time / Historic’ and ‘International’. 

Ended up reducing to ~14,800 tracks, covering 12 genres (12 class problem)

## Librosa
Librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems. Includes capabilites for :
 - Visualisation
 - Audio Playback
 - Feature Extraction (Spectral, Rhythm , Onset Detection, beats and tempo etc.) 

Python package Librosa can be found [here](https://librosa.org/doc/latest/index.html)

## Dimensionality reduction techniques
Since extracted features from Librosa are composed of high dimensional arrays we cannot model on these as individual features due to limited processing power and disk space. Needed to process further with dimensionality reduction techniques. I employed the following approaches :
 1. Process extracted features with summary statistics (SS)
 2. Principal component analysis  on extracted Features (PCA)

## Exploratory Data Analysis (EDA)
![EDA 1 ](https://user-images.githubusercontent.com/74214807/112044559-3ea87400-8b4a-11eb-9c40-1dd8aa46238f.jpg)
![EDA2](https://user-images.githubusercontent.com/74214807/112044605-4831dc00-8b4a-11eb-87da-ce000441dc1d.jpg)




**Model  | Scores**
------------- | -------------
Random Forest  | 0.64
SVM  | 0.61
KNN | 0.58

![furtherEDA2](https://user-images.githubusercontent.com/74214807/112044582-449e5500-8b4a-11eb-8fa5-b1c20b6eed3e.jpg)
![furtherEDA1](https://user-images.githubusercontent.com/74214807/112044586-45cf8200-8b4a-11eb-9dec-4b054bdfe60e.jpg)
![results1](https://user-images.githubusercontent.com/74214807/112044591-46681880-8b4a-11eb-9e7b-67062ccc9e75.jpg)
![method2](https://user-images.githubusercontent.com/74214807/112044594-4700af00-8b4a-11eb-8109-af871efde358.jpg)
![method1](https://user-images.githubusercontent.com/74214807/112044597-4700af00-8b4a-11eb-9cab-3e00a1628700.jpg)
![EDA5](https://user-images.githubusercontent.com/74214807/112044600-47994580-8b4a-11eb-89b0-a4914d9c6233.jpg)
![EDA4](https://user-images.githubusercontent.com/74214807/112044602-47994580-8b4a-11eb-8cae-c4a291f7dd21.jpg)
![EDA3](https://user-images.githubusercontent.com/74214807/112044603-4831dc00-8b4a-11eb-8308-0abe3e760fe4.jpg)


