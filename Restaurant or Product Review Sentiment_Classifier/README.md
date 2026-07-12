# Restaurant or Product Review Sentiment Classifier

This project performs binary sentiment analysis on customer review text. It uses a dataset of review ratings and text comments to classify whether a review is positive or negative.

## Project Workflow

1. Load and inspect the dataset from `Reviews.csv`.
2. Remove neutral 3-star reviews.
3. Convert ratings to binary sentiment labels: positive for scores 4 and 5, negative for scores 1 and 2.
4. Preprocess review text using stopword removal, stemming, and tokenization.
5. Transform text into numeric features using `CountVectorizer`.
6. Split the data into training and test sets.
7. Train and evaluate three classifiers:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Artificial Neural Network (ANN)

## Important Notes

- The dataset is imbalanced: positive reviews make up the majority of labels.
- The ANN model uses `class_weight` to account for the minority negative class.

## Files

- `project.ipynb` ? notebook containing code and exploratory analysis.
- `Reviews.csv` ? dataset file (excluded from Git tracking to avoid pushing large data).

