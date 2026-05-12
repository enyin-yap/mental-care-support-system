from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChatbotTrainer:
    def __init__(self, train_df, test_df, model_save_path='chatbot_model.h5',
                 batch_size=32, max_length=100, epochs=15):
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.epochs = epochs
        # Prepare data
        self.train_df = train_df
        self.test_df = test_df

        # Initialize tokenizer and label encoder
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()

        # Prepare the data
        self._prepare_data()

        # Build the model
        self._build_model()

    def _prepare_data(self):
        # Debug: Print sample inputs before tokenization
        logging.info("Sample inputs before tokenization:")
        logging.info(self.train_df['input'].head())

        # Fit tokenizer on training data
        self.tokenizer.fit_on_texts(self.train_df['input'])

        # Debug: Print vocabulary details
        vocab = self.tokenizer.word_index
        logging.info(f"Total vocabulary size: {len(vocab)}")
        logging.info("Sample vocabulary items:")
        sample_words = list(vocab.items())[:10]
        for word, index in sample_words:
            logging.info(f"Word: {word}, Index: {index}")

        # Convert text to sequences
        X_train = self.tokenizer.texts_to_sequences(self.train_df['input'])
        X_test = self.tokenizer.texts_to_sequences(self.test_df['input'])

        # Debug: Print sequence details
        logging.info(f"Sample sequence from X_train: {X_train[0]}")

        # Pad sequences
        self.X_train = pad_sequences(X_train, maxlen=self.max_length)
        self.X_test = pad_sequences(X_test, maxlen=self.max_length)

        # Debug: Print padded sequence details
        logging.info(f"Sample padded sequence: {self.X_train[0]}")

        # Encode labels
        self.label_encoder.fit(pd.concat([self.train_df['intent'], self.test_df['intent']]))
        self.y_train = self.label_encoder.transform(self.train_df['intent'])
        self.y_test = self.label_encoder.transform(self.test_df['intent'])

        # Debug: Print label details
        logging.info(f"Number of unique responses: {len(self.label_encoder.classes_)}")
        logging.info("Sample responses and their encodings:")
        for i in range(min(5, len(self.label_encoder.classes_))):
            logging.info(f"Response: {self.label_encoder.classes_[i]} -> Encoding: {i}")

        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.num_classes = len(self.label_encoder.classes_)

    def _build_model(self):
        self.model = models.Sequential([
            layers.Embedding(self.vocab_size, 128, input_length=self.max_length),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logging.info(self.model.summary())

    def train(self):
        logging.info("Starting training...")

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )

        # Save the model and tokenizer
        self.save_model()

        return history

    def save_model(self):
        # Save the model
        self.model.save(self.model_save_path)
        logging.info(f"Model saved to {self.model_save_path}")

        # Save the tokenizer
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the label encoder
        with open('label_encoder.pickle', 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("Tokenizer and label encoder saved")


def main():
    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        from data_preprocess import combine_all_datasets
        combined_dataframe = combine_all_datasets()

        print(combined_dataframe.head(100))

        # Split the data
        logging.info("Splitting data into train and test sets...")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(combined_dataframe, test_size=0.2, random_state=42)

        # Initialize trainer
        trainer = ChatbotTrainer(
            train_df=train_df,
            test_df=test_df,
            batch_size=16,  # Conservative batch size for CPU
            max_length=64,  # Balanced sequence length
            epochs=15
        )

        # Train the model
        trainer.train()

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

# Reference and online sources used in this seciton of the code
#
