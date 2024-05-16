"""Script to create a classifier for reddit comments."""

# Import libraries
import os
import time
import json
import logging
import warnings
import psycopg2
import pandas.io.sql as sqlio
import tiktoken
from openai import OpenAI, RateLimitError

# Set configurations
warnings.filterwarnings("ignore")
logging.basicConfig(
    level = logging.INFO, format = '%(levelname)s:%(asctime)s:%(message)s'
)

# Load environment variables
logging.info("Loading environment variables")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NUMBER_OF_LABELLED_DATA = 300
MODEL = "gpt-4o"

# Load data from the database
logging.info("Loading data from the database")
connection = psycopg2.connect(
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)
data = sqlio.read_sql_query(
    "SELECT * FROM reddit_usernames_comments;", connection
)
connection.close()

# Filter comments that have more than 1000 tokens to avoid OpenAI API errors
logging.info("Filtering comments with more than 1000 tokens")
def get_num_tokens(text, model="gpt-4o"):
    """Function to return the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    num_util_tokens = 9
    num_tokens = len(encoding.encode(text))
    num_tokens += num_util_tokens
    return num_tokens


data["num_of_tokens"] = data["comments"].apply(get_num_tokens)
data = data.loc[data["num_of_tokens"] <= 1000]

# Get a smaller subset of the data to label
logging.info("Sampling the dataset")
data = data.sample(n=NUMBER_OF_LABELLED_DATA, random_state=22)
data.reset_index(drop=True, inplace=True)
data.drop(columns=["num_of_tokens"], inplace=True)

# Initialize OpenAI client and load the prompt template
client = OpenAI(api_key=OPENAI_API_KEY)
logging.info("Loading the prompt template")
with open("prompt_template.txt", "r", encoding="utf-8") as prompt_file:
    prompt_template = prompt_file.read()

# Define parameters for an imaginary function in OpenAI function calling format.
# Function calling is used because it forces the model to output the value
# without any other unnecessary information.
function_tools = [
    {
        "type": "function",
        "function": {
            "name": "process_profession",
            "description": (
                "Function to process the profession of the comment's writer"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profession": {
                        "type": "string",
                        "description": (
                            "Profession of the writer of the comment. Allowed "
                            "values are 'Medical doctor', 'Veterinarian', and "
                            "'other'."
                        ),
                    }
                },
                "required": ["profession"],
            },
        },
    }
]

# Query OpenAI API to label the data
labels = []
for index, row in data.iterrows():

    logging.info(
        "Labelling comment %s of %s", index + 1, NUMBER_OF_LABELLED_DATA
    )
    prompt = prompt_template.format(comment=row["comments"])

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            tools=function_tools,
            tool_choice={
                "type": "function",
                "function": {"name": "process_profession"},
            },
        )
    except RateLimitError as error:
        logging.warning("Rate limit error. Waiting for 60 seconds.")
        time.sleep(60)
        logging.info("Retrying the request")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            tools=function_tools,
            tool_choice={
                "type": "function",
                "function": {"name": "process_profession"},
            },
        )

    labels.append(
        json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )["profession"].lower()
    )

data["labels"] = labels

# Save the labelled data
logging.info("Saving the labelled data")
data.to_csv("data/data_generated.csv", index=False)
