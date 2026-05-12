import pandas as pd
import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Data preprocessing class for dataset 1
class PreprocessDataHuggingFace:
    def __init__(self, dataset_name, test_size=0.2, random_state=42, max_seq_length=20):
        self.dataset_name = dataset_name
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.encoder = LabelEncoder()
        self.test_size = test_size
        self.random_state = random_state
        self.max_seq_length = max_seq_length
        self.intent_keywords = {
            "depression": ["depression","depressed", "sad", "low", "down", "hopeless", "gloomy", "despair", "worthless", "empty","dysthymia","antidepressant", "antidepressants"],
            "anxiety": ["anxiety", "worried", "nervous", "panic", "fear", "overwhelmed", "stress", "anxious", "panic attack"],
            "suicide": ["suicidal", "kill myself", "harm myself", "end my life", "i dont want to live", "hopeless","suicide"],
            "mental_health_info": ["family history", "mental","illness", "mentalillness", "immune", "genetics", "risk", "heredity", "genetic predisposition", "family background", "genetic factors","medication", "unwell", "meds","therapy","support group", "therapist","psychiatrist","psychologist","counselling"," psychotherapy","hypnotherapist","selfmanagement","sleep", "hypnotherapy","neurofeedback","biofeedback","insomnia"],
            "racism": ['racism', 'racial injustice', 'discrimination', 'racist', 'race inequality', 'racism awareness', 'equality', 'social justice','racial trauma','racial'],
            "schizophrenia": ["schizophrenia", "delusions", "hallucinations", "psychosis", "paranoia", "disorganized speech", "hearing voices", "mental break", "split personality"],
            "addiction": ["addiction","addictions", "drug addiction", "alcoholism", "substance abuse", "dependency", "withdrawal", "craving", "rehab", "sobriety", "addicted", "overdose", "numb", "relapse","smoking","alcohol"],
            "general_inquiry": ["help", "support", "information", "question", "how", "why","who", "what", "tell me more", "curious", "can you explain", "can you help", "assist", "need advice", "I have a question"],
            "goodbye":["bye", "see you later", "goodbye", "take care","i need to go", "talk to you later", "have a good day","thanks, bye", "that's all for now", "until next time","i should get going", "bye for now"]
        }

    def load_data(self):
        hf_ds = load_dataset(self.dataset_name)
        self.df = hf_ds['train'].to_pandas()

    def clean_data(self, text):
        # Convert to lowercase
        text = text.lower()
        # Replace newlines with spaces and strip extra spaces
        text = text.replace('\n', ' ').strip()
        # Remove special characters (only alphanumeric and spaces)
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text

    def split_input_response(self, text):
        try:
            question = text.split("human")[1].split("assistant")[0].strip()
            answer = text.split("assistant")[1].strip()
            return question, answer
        except IndexError:
            return None, None

    def preprocess(self):
        # Clean the 'text' column
        self.df['text'] = self.df['text'].apply(self.clean_data)

        # Split the text into input and response
        self.df[['input', 'response']] = self.df['text'].apply(
            lambda x: pd.Series(self.split_input_response(x))
        )

        # Drop the 'text' column as it's no longer needed
        self.df.drop(columns=['text'], inplace=True)

        # Tokenize and lemmatize the input and response text
        self.df['input_tokens'] = self.df['input'].apply(self.tokenize_and_lemmatize)
        self.df['response_tokens'] = self.df['response'].apply(self.tokenize_and_lemmatize)

        self.generate_intents()

        # Encode the responses as labels
        # Reference: https://medium.com/@kattilaxman4/a-practical-guide-for-python-label-encoding-with-python-fb0b0e7079c5
        self.df['response_encoded'] = self.encoder.fit_transform(self.df['response'])

        # Split into train and test data
        self.split_data()

    def generate_intents(self):
        self.df['intent'] = self.df['input'].apply(self.assign_intent)

    def assign_intent(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens]

        # Check which intent category the text belongs to based on keywords
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in lemmatized_tokens for keyword in keywords):
                return intent
        return "unknown"  # If no intent is found

    # Reference: https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/
    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def split_data(self):
        X = self.df['input_tokens'].apply(lambda x: ' '.join(x))
        y = self.df['response_encoded'].tolist()

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Store the split data
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_preprocessed_data(self):
        return self.df


# Data preprocessing class for dataset 2
class PreprocessDataHuggingFace2:
    def __init__(self, dataset_name, test_size=0.2, random_state=42, max_seq_length=20):
        self.dataset_name = dataset_name
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.encoder = LabelEncoder()
        self.test_size = test_size
        self.random_state = random_state
        self.max_seq_length = max_seq_length

    def load_data(self):
        hf_ds = load_dataset(self.dataset_name)
        self.df = hf_ds['train'].to_pandas()

    def clean_data(self, text):
        if not isinstance(text, str):
          return ""
        # Convert to lowercase
        text = text.lower()
        # Replace newlines with spaces and strip extra spaces
        text = text.replace('\n', ' ').strip()
        # Remove special characters (only alphanumeric and spaces)
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text


    def preprocess(self):

        # Rename the column and drop unnecessary columns
        self.df = self.df[['questionText', 'answerText', 'topic']].rename(columns={'questionText': 'input', 'answerText': 'response', 'topic': 'intent'})

        # Clean the text
        self.df['input'] = self.df['input'].apply(self.clean_data)
        self.df['response'] = self.df['response'].apply(self.clean_data)

        # Tokenize and lemmatize the input and response text
        self.df['input_tokens'] = self.df['input'].apply(self.tokenize_and_lemmatize)
        self.df['response_tokens'] = self.df['response'].apply(self.tokenize_and_lemmatize)


        # Encode the responses as labels
        self.df['response_encoded'] = self.encoder.fit_transform(self.df['response'])

        # Split into train and test data
        self.split_data()


    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def split_data(self):
        X = self.df['input_tokens'].apply(lambda x: ' '.join(x))
        y = self.df['response_encoded'].tolist()

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Store the split data
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_preprocessed_data(self):
        return self.df

# Data preprocessing class for dataset 3
class PreprocessDataHuggingFace3:
    def __init__(self, dataset_name, test_size=0.2, random_state=42, max_seq_length=20):
        self.dataset_name = dataset_name
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.encoder = LabelEncoder()
        self.test_size = test_size
        self.random_state = random_state
        self.max_seq_length = max_seq_length
        self.intent_keywords = {
            "autism": ["autism", "autistic", "autistic spectrum", "asd", "autism spectrum disorder", "sensory overload", "social communication", "repetitive behavior", "fixated interests", "social difficulties", "nonverbal", "early intervention", "diagnosed with autism", "sensory processing disorder", "spectrum disorder"],
            "adhd": ["adhd", "attention deficit", "hyperactivity", "inattention", "impulsive", "impulsivity", "fidgeting","concentration", "focus issues", "difficulty concentrating", "attention deficit hyperactivity disorder", "restlessness", "difficulty staying still", "distractible", "impulse control",'sugar'],
            "down_syndrome":['down','downsyndrome','genetic','chromosome','choromosomes'],
            "developmental_delay": ["developmental delay",'delay', "delayed development", "speech delay", "motor delay", "cognitive delay", "early intervention", "learning delay", "delayed milestones", "developmental","milestones", "speech therapy", "occupational therapy", "physical therapy", "special education", "early childhood intervention"],
            "child_development": ["child","toddler", "toddler behavior", "toddler milestones", "milestones", "toddler development", "toddler speech", "toddler growth","tantrum", "child tantrum", "toddler tantrum", "temper tantrum", "emotional outburst", "child behavior", "managing tantrums","sleep training", "sleep habits", "baby sleep", "sleep schedule", "sleep training tips", "sleep regression","potty training", "toilet training", "training a toddler", "potty schedule", "potty training tips", "diaper"],
            "neurodivergent": ["neurodivergent",'nt','nd','neurodivergence','neurodiversity'],
            "celebral_palsy": ["celebralpalsy",'celebral','palsy','cp']
        }


    def load_data(self):
        hf_ds = load_dataset(self.dataset_name)
        self.df = hf_ds['train'].to_pandas()

    def clean_data(self, text):
        if not isinstance(text, str):
          return ""
        # Convert to lowercase
        text = text.lower()
        # Replace newlines with spaces and strip extra spaces
        text = text.replace('\n', ' ').strip()
        # Remove special characters (only alphanumeric and spaces)
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text


    def preprocess(self):

        # Concatenate & Rename the column and drop unnecessary columns
        self.df['input'] = self.df['question '] + ' ' + self.df['Q1-Knowledge on Topic '] + ' ' + self.df['Q2-User Intent and Context'] + ' '+ self.df['Q3-User details']
        self.df = self.df.rename(columns={'Output': 'response'})
        self.df = self.df[['input', 'response']]

        # Clean the text
        self.df['input'] = self.df['input'].apply(self.clean_data)
        self.df['response'] = self.df['response'].apply(self.clean_data)

        # Tokenize and lemmatize the input and response text
        self.df['input_tokens'] = self.df['input'].apply(self.tokenize_and_lemmatize)
        self.df['response_tokens'] = self.df['response'].apply(self.tokenize_and_lemmatize)

        self.generate_intents()

        # Encode the responses as labels
        self.df['response_encoded'] = self.encoder.fit_transform(self.df['intent'])

        # Split into train and test data
        self.split_data()

    def generate_intents(self):
        self.df['intent'] = self.df['input'].apply(self.assign_intent)

    def assign_intent(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens]

        # Check which intent category the text belongs to based on keywords
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in lemmatized_tokens for keyword in keywords):
                return intent
        return "unknown"  # If no intent is found



    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def split_data(self):
        X = self.df['input_tokens'].apply(lambda x: ' '.join(x))
        y = self.df['response_encoded'].tolist()

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Store the split data
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_preprocessed_data(self):
        return self.df

# Data preprocessing class for dataset 4
class PreprocessDataHuggingFace4:
    def __init__(self, dataset_name, test_size=0.2, random_state=42, max_seq_length=20):
        self.dataset_name = dataset_name
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.encoder = LabelEncoder()
        self.test_size = test_size
        self.random_state = random_state
        self.max_seq_length = max_seq_length

    def load_data(self):
        hf_ds = load_dataset(self.dataset_name)
        self.df = hf_ds['train'].to_pandas()

    def clean_data(self, text):
        if not isinstance(text, str):
          return ""
        # Convert to lowercase
        text = text.lower()
        # Replace newlines with spaces and strip extra spaces
        text = text.replace('\n', ' ').strip()
        # Remove special characters (only alphanumeric and spaces)
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text


    def preprocess(self):

        # Rename the column and drop unnecessary columns
        self.df = self.df[['Context', 'Response', 'Knowledge']].rename(columns={'Context': 'input', 'Response': 'response', 'Knowledge': 'intent'})
        self.df['intent'] = self.df['intent'].replace('thanks', 'gratitude')

        # Clean the text
        self.df['input'] = self.df['input'].apply(self.clean_data)
        self.df['response'] = self.df['response'].apply(self.clean_data)

        # Tokenize and lemmatize the input and response text
        self.df['input_tokens'] = self.df['input'].apply(self.tokenize_and_lemmatize)
        self.df['response_tokens'] = self.df['response'].apply(self.tokenize_and_lemmatize)


        # Encode the responses as labels
        self.df['response_encoded'] = self.encoder.fit_transform(self.df['response'])

        # Split into train and test data
        self.split_data()


    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def split_data(self):
        X = self.df['input_tokens'].apply(lambda x: ' '.join(x))
        y = self.df['response_encoded'].tolist()

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Store the split data
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_preprocessed_data(self):
        return self.df

# Combine all datasets into single DataFrame
def combine_all_datasets():
    # Initialize the classes with the dataset names
    preprocessors = [
        PreprocessDataHuggingFace("heliosbrahma/mental_health_chatbot_dataset"),
        PreprocessDataHuggingFace2("nbertagnolli/counsel-chat"),
        PreprocessDataHuggingFace3("Indulekha/mental-health"),
        PreprocessDataHuggingFace4("alexandreteles/mental-health-conversational-data"),
    ]

    processed_dataframes = []

    for preprocessor in preprocessors:
        preprocessor.load_data()
        preprocessor.preprocess()
        processed_df = preprocessor.get_preprocessed_data()
        processed_dataframes.append(processed_df)

    # Combine all preprocessed DataFrames
    combined_df = pd.concat(processed_dataframes, ignore_index=True)

    # Add these debug prints using the correct variable name
    print("Dataset size:", len(combined_df))
    print("Sample data:")
    print(combined_df.head())

    # Verify data isn't empty
    if combined_df.empty:
        raise ValueError("Combined dataframe is empty!")

    return combined_df



# Call the function
combined_df = combine_all_datasets()
print(combined_df.head())
