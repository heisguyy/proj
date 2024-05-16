"""Script to train the classifier for reddit comments."""

# Import necessary libraries libraries
import os
import logging
import warnings

import joblib
import numpy as np
import psycopg2
import pandas.io.sql as sqlio
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Set configurations
np.random.seed(13)
warnings.filterwarnings("ignore")
logging.basicConfig(
    level = logging.INFO, format = '%(levelname)s : %(asctime)s : %(message)s'
)

# Load environment variables
logging.info("Loading environment variables")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# load the data
logging.info("Loading the data")
data = pd.read_csv("data/training_data.csv")

# Initialize label encoder and convert labels to numerical values
logging.info("Converting labels to numerical values")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data.labels)

# Initialize bag of words encoder and extract numerical features from comments
logging.info("Extracting numerical features from comments")
bag_of_words_encoder = CountVectorizer(stop_words="english")
features = bag_of_words_encoder.fit_transform(data.comments)

# Initialize and train the model on all data
logging.info("Training the model")
params = {
    "max_depth": 11,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "criterion": "log_loss"
}
model = DecisionTreeClassifier(**params)
model.fit(features, labels)

# Save the artifacts
logging.info("Saving the artifacts")
joblib.dump(label_encoder, "artifacts/label_encoder.joblib")
joblib.dump(bag_of_words_encoder, "artifacts/bag_of_words_encoder.joblib")
joblib.dump(model, "artifacts/classifier.joblib")

# Load data from the database
logging.info("Loading test data from the database")
connection = psycopg2.connect(
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)
test_data = sqlio.read_sql_query(
    "SELECT * FROM reddit_usernames_comments;", connection
)
connection.close()

# Predict categories for each comment
logging.info("Preparing test data.")
test_features = bag_of_words_encoder.transform(test_data["comments"])
test_data["labels"] = model.predict(test_features)
test_data["labels"] = label_encoder.inverse_transform(test_data["labels"])

# Saved predictions
logging.info("Saving predictions.")
test_data.to_csv("data/predictions.csv", index=False)
