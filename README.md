# Music Genre Classification
## Introduction
This project was completed during my time on the General Assembly Data Science Immersive course.

Through my analysis of the Free Music Archive audio files and use of the Python package, Librosa. I aim to determine the class of music genre of an audio file. As well as understand the key predictors of this type of classification problem.

I performed various statistical and machine learning models for the classification problem including one hyperparameter grid search on AWS. The model that generalises best was a Random Forest Classifier, predicting the music genre with a 65% accuracy compared to the baseline of 30%. Key insights and limitations in accuracy were found, such as increase of bias from class imbalance, inaccurate tagging of metadata for model training, limit of explained variance due to required dimensionality reduction.

Future improvements and general limitations have been summarised below.
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
Put very simply the main question I am asking - “Is there a way to automate music genre classification?”. Ultimately the current norm is still to manually assign genre labels leading to slow and tedious tagging methods. Automating this procedure is crucial for large music databases and music information retrieval systems  as the music industry accelerates into the digital realm.

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
Some of the brief data analysis of classes and particular features shown here : 
<img src="https://user-images.githubusercontent.com/74214807/112045733-782daf00-8b4b-11eb-9822-fda8ff104ef9.png" width="566" height="308" />
<img src="https://user-images.githubusercontent.com/74214807/112045710-7368fb00-8b4b-11eb-8039-b367239e3b2e.png" width="685" height="358" />

Full EDA for either method found here : [SS]() and [PCA]()

## Methodology
Given both dimensionality reduction techniques employed I have summarised the general methodologies for either approach below :

 ### 1. Summary Statistics (SS)
  - After extracting the desired features, get summary statistics for each feature (mean, median, std, min, max, kurtosis, skew)
  - Calculate each statistic on'b' a total of 'a' times for a given feature array with shape : (a , b) 
  - For each track, store each statistic for a given feature in a dictionary (dictionary within dictionary)
  - Model on summary statistics
<img src="https://user-images.githubusercontent.com/74214807/112047537-b926c300-8b4d-11eb-882f-32b5c51d6079.png" width="814" height="333" />

 ### 2. Principal Component Analysis (PCA)
  - Created a function which takes the ATS (amplitude spectogram, y) for each track_id
  - Calculate the Decibel Scaled Spectogram, DB for each track_id
  - Apply PCA (chose number of components such that total explained variance was 97%)
  - Model on PCA components
<img src="https://user-images.githubusercontent.com/74214807/112047988-3ce0af80-8b4e-11eb-82d9-954df2b11628.png" width="814" height="351" />

## Modelling
For the multiclass classification problem I selected the following classifiers :
 - Random Forest
 - SVM
 - KNN
 - Logistic Regression
 - Decision Tree
 - Adaboost (Decision Tree as base estimator)
 - Naive Bayes
 - Bagging

## Results

After hyperparameter tuning with various GridSearches, the model with the highest CV score was found to be a Random Forest classifier. 
The top 3 accuracy scores for 12 unbalanced genre classes :

**Model**  | **Scores**
------------- | -------------
Random Forest  | 0.64
SVM  | 0.61
KNN | 0.58

We are able to identify a handful of the most important ones but not to a highly significant degree to have them stand out from the other indicators. The most reaccuring important features at the top of the list were 'H', 'P', 'melspec' and 'contrast' which denote the harmonic and or percussive elements of the track as well the dynamic of the spectral range and their difference. These are some of the most intuitive criterias that one could perceive objectively to distinguish between genres. (spectral peak, valley, and their difference in each frequency subband)

As seen below these 'H', 'P' and spectral features seem to distinguish certain genres relatively well.

<img src="https://user-images.githubusercontent.com/74214807/112135440-0ea2b480-8bce-11eb-99f3-769e9bdd7203.png" width="779" height="407" />

## Key learnings
 - Audio features high dimensionality : When dealing with audio features the high dimensionality of the extracted raw features limits the number of features that can be modelled on. Moreover, the high dimensionality requires reduction techniques to be employed limiting the variance these models can train on. 
 - Inaccurately tagged meta-data : Having inaccurately tagged meta-data regarding a tracks' genre leads to a wrong representation of the data to be modelled on, increasing false positives and false negatives, overall reducing the model accuracy. 
 - Unknown classes : Equivalently, having other established genres in the dataset will also diminish model performance in a similar manner. 
 - Class Imbalance : The large class imbalance increases bias as machine learning classifiers tend to be more biased towards majority class, badly classifying minority classes. In the future, need to increase dataset and adjust class supports to reach a balanced data pool to train on.

Labeling music into genres is arbitrary, and the line between one genre and another is more often than not, blurred. As if it has been formed from several ‘flavours’ of genres, with a set of sub-genres and a type of ‘style’. Genre classification can be quite subjective and not so simple as classifying a colour, with a certain measurable wavelength. However, there are perceptual criteria that are related to instrumentation, structure of the rhythm, harmonics and texture of the music that can play a role in characterising a particular genre. Methods for automated genre classification would add value to many music information retrival systems, music apps and music streaming platforms of which many are still manually labelled.

## Dependencies
See the requirements.txt file specific dependencies

 NumPy\
 Pandas\
 Matplotlib\
 Seaborn\
 Pickle\
 Scikit-learn\
 Librosa

## License
Copyright © ISMIR, 2000-2021



