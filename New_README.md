#StreamOracle

##Overview:
This project aims to predict the quality and popularity for a movie or TV show, based on features like the cast, plot summary, genres etc. The intent is to help streaming platforms decide whether or not to greenlight/acquire a new piece of content. The quality and popularity are measured by IMDB rating and number of IMDB votes respectively. (Note that this project is relatively unique because there’s 2 target variables.)

This is important because streaming companies are under a lot of financial pressure in 2023. Netflix hit its all-time market cap high in Nov 2021 (\$314 Bn) driven by pandemic-related lockdowns but has since plummeted to $145 Bn. It’s facing subscriber loss and has announced plans to launch a lower-priced, ad-supported tier. Part of the reason is more competition in a now-crowded streaming space. Another reason is that companies (across the tech sector) bet on pandemic-era trends being more permanent, which didn’t necessarily happen.
Whatever the reasons, streamers now have a much smaller margin for error in their selection of projects. This project aims to help them make better decisions.

##Background on the subject matter:
Given the right data, it would be entirely possible to predict the success of a movie or TV show. Studios are already doing this - have a look at this Verge article about [LA based start-up Cinelytic] (https://www.theverge.com/2019/5/28/18637135/hollywood-ai-film-decision-script-analysis-data-machine-learning) that is advising Hollywood on which movies to make.

##Notebooks:

**1 - Capstone**:
This is the notebook kicking off the project. It has all the data cleaning, EDA and feature engineering.

**2 - Awards Data**:
This is supplementary data on the Oscar and Emmy awards that I thought might be useful, which I decided to add in later.

**3 - IMDB Votes Modeling**:
Our first modelling notebook, where I try different models for the regression problem IMDB votes

**4 - IMDB Score Modeling**:
Our second modelling notebook, where I try different models for the classification problem IMDB Score

**5 - Neural Networks with Word Embeddings**:
An alternate approach to modelling and feature engineering, using word embeddings on the 2 text columns (Description and Titles) and then trying neural networks.


##Recreating the conda environment:
The environment set-up is in the requirements.txt file. You can follow input the commands in terminal below to re-create it:
- Create a new conda environment: `conda create -n new_environment_name python=your_python_version`
- Activate the new environment: `conda activate new_environment_name`
- c. Install the packages from the requirements.txt file: `pip install -r requirements.txt`

##Data:
All the data is available in the folder 'final data'. The links to the original source data is:
- [Netflix](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- [Prime Video](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows)
- [Disney Plus](https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows)
- HBO Max dataset has since been taken down for some reason, but the data is available in the folder
- [Oscar Winners and Nominees](https://www.kaggle.com/datasets/unanimad/the-oscar-award)
- [Emmy Winners and Nominees](https://www.kaggle.com/datasets/unanimad/emmy-awards)