# Ensure model_training.py has been run to generate local .h5 and .pkl files before executing this script.

import pickle

# Load your encoder to see what the AI learned
with open('label_encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
    print("The AI knows these intents:")
    print(encoder.classes_)