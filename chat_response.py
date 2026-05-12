import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Dict, Any, Optional
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseHandler:
    def __init__(self,
                 model_path: str = 'chatbot_model.h5',
                 tokenizer_path: str = 'tokenizer.pickle',
                 encoder_path: str = 'label_encoder.pickle',
                 max_length: int = 64,
                 confidence_threshold: float = 0.1):
        """
        Initialize the response handler with necessary components.

        Args:
            model_path: Path to the trained TensorFlow model
            tokenizer_path: Path to the saved tokenizer
            encoder_path: Path to the saved label encoder
            max_length: Maximum length of input sequences
            confidence_threshold: Minimum confidence score for responses
        """
        self.intent_responses = {
            # --- Greetings & Casual ---
            "greeting": "Hello! I'm here to listen. How are you feeling today?",
            "morning": "Good morning. I hope your day is off to a manageable start. I'm here if you need to talk.",
            "afternoon": "Good afternoon. How has your day been so far?",
            "evening": "Good evening. I'm here to support you before you head to rest.",
            "night": "It's late, but I'm still here for you. What's on your mind?",
            "casual": "I'm here for you, whether you want to talk about big things or just chat.",
            "goodbye": "Take care of yourself. I'm here whenever you need to talk again.",
            "done": "I understand. I'll be here if you need to return to this conversation later.",
            "gratitude": "You're very welcome. I'm glad I could be here for you.",
            "happy": "It's wonderful to hear that you're having a positive moment! Tell me more about it.",

            # --- Sadness & Self-Worth ---
            "sad": "I'm sorry you're feeling down. I'm here to listen to whatever is making your heart heavy.",
            "depressed": "I hear how much pain you're in. Depression can feel like a heavy weight, but you don't have to carry it alone.",
            "depression": "I hear how much pain you're in. It's okay to feel this way, and I'm here for you.",
            "worthless": "I want you to know that you have inherent value, even when it's hard to see it. Your feelings matter.",
            "self-esteem": "Building self-love takes time. We can talk about the things that are making you doubt yourself.",
            "lonely": "You're not alone. I'm here to listen whenever you need a safe space to talk.",

            # --- Anxiety, Stress & Overwhelm ---
            "anxiety": "It sounds like things are feeling overwhelming right now. Let's try to take a slow breath together.",
            "anxious": "It sounds like you're feeling a lot of pressure. I'm here to help you sit through this feeling.",
            "stressed": "Stress can be so heavy. What's been the biggest challenge you've faced today?",
            "stress": "I hear the stress in your words. Let's try to break things down into smaller pieces together.",
            "scared": "It's okay to feel afraid. I'm right here with you. What is making you feel unsafe or worried?",
            "trauma": "I'm so sorry you've been through something so difficult. Please be gentle with yourself as we talk.",

            # --- Specific Conditions & Neurodivergence ---
            "adhd": "Managing focus and energy can be exhausting. I'm here if you need to vent about the frustration.",
            "autism": "The world can be overwhelming sometimes. I'm a safe space for you to express yourself exactly as you are.",
            "neurodivergent": "I appreciate you sharing that with me. How can I best support your specific needs today?",
            "addiction": "Recovery is a journey with many ups and downs. I'm here to support you without judgment.",
            "substance-abuse": "Thank you for being honest. Are you currently in a safe place or working with a support group?",
            "eating-disorders": "I hear the struggle you're describing. Food and body image issues are incredibly tough to navigate.",

            # --- Relationships & Family ---
            "relationships": "Relationships can be our greatest joy and our biggest stress. What's happening with yours?",
            "family-conflict": "Family dynamics are complicated. I'm here if you need to vent about what's happening at home.",
            "marriage": "Marriage comes with unique challenges. I'm listening if you need to talk about your partner.",
            "domestic-violence": "I am very concerned for your safety. Please know that there are people who can help you find a safe place.",

            # --- Crisis & Safety (High Priority) ---
            "suicide": "I am very concerned about your safety. Please contact Talian HEAL (15555) or go to the nearest emergency room immediately.",
            "self-harm": "I am very concerned about your safety. Please reach out to a professional or a crisis hotline (15555) immediately.",

            # Information & General Inquiry 
            "mental_health_info": "I can provide general information about mental health, but remember I'm not a doctor. What would you like to know?",
            "learn-more": "I'm glad you're interested in learning. Knowledge is a great tool for mental wellness. What area should we explore?",
            "general_inquiry": "I'm here to provide support and information. How can I help you today?",
            "understand": "I'm glad we're connecting. I'm trying my best to understand your experience.",

            # Fallbacks 
            "unknown": "I'm not entirely sure I understand, but I'm here for you. Could you tell me more about that?",
            "default": "I'm here to support you. Could you rephrase that so I can understand better?"
        }

        try:
            # Load the trained model and required components
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.max_length = max_length
            self.confidence_threshold = confidence_threshold

            # Load tokenizer for text preprocessing
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)

            # Load label encoder for response classification
            with open(encoder_path, 'rb') as handle:
                self.label_encoder = pickle.load(handle)

            logger.info("Response handler initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing response handler: {str(e)}")
            raise

    def preprocess_input(self, text: str) -> np.ndarray:
        """
        Clean and prepare the input text for the model.

        Args:
            text: Raw input text from user

        Returns:
            Preprocessed and padded sequence ready for model input
        """
        text = text.lower().strip()
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_length)
        return padded

    def get_top_responses(self, prediction: np.ndarray, n: int = 3) -> list:
        """
        Get the top N most likely responses with their confidence scores.

        Args:
            prediction: Model's output predictions
            n: Number of top responses to return

        Returns:
            List of (response, confidence) tuples
        """
        top_indices = np.argsort(prediction[0])[-n:][::-1]
        return [
            (self.label_encoder.inverse_transform([idx])[0], float(prediction[0][idx]))
            for idx in top_indices
        ]

    def get_response(self, user_input: str) -> Dict[str, Any]:
        """
        Generate a response for the user input.

        Args:
            user_input: User's input text

        Returns:
            Dictionary containing response details including:
            - status: success/error
            - response: primary response text
            - confidence: confidence score
            - alternative_responses: other possible responses
            - low_confidence: flag for low confidence responses
        """
        try:
            # Preprocess and get model prediction
            processed_input = self.preprocess_input(user_input)
            prediction = self.model.predict(processed_input, verbose=0)

            # Get top responses
            top_responses = self.get_top_responses(prediction)
            predicted_intent, confidence = top_responses[0]

            # Final message shown to user
            if confidence < self.confidence_threshold:
                final_message = self.intent_responses["unknown"]
                is_low_conf = True
            else:
                # Look up the predefined message for this intent
                final_message = self.intent_responses.get(predicted_intent, self.intent_responses["unknown"])
                is_low_conf = False

            # Prepare response data
            response_data = {
                'status': 'success',
                'response': final_message,
                'confidence': confidence,
                'intent': predicted_intent,
                'low_confidence': is_low_conf
            }

            # Add fallback message for low confidence responses
            if response_data['low_confidence']:
                response_data['fallback_message'] = (
                    "I'm not entirely sure about this response. "
                    "Would you like to rephrase your question?"
                )

            return response_data

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'status': 'error',
                'response': "I apologize, but I'm having trouble processing your request.",
                'error': str(e),
                'confidence': 0.0
            }

    def validate_input(self, user_input: str) -> Optional[str]:
        """
        Check user input for potential issues.

        Args:
            user_input: User's input text

        Returns:
            Error message if validation fails, None if input is valid
        """
        if not user_input.strip():
            return "Please provide a non-empty input."
        if len(user_input) > 500:
            return "Input is too long. Please limit your message to 500 characters."
        return None

# below are references that have looked through while doing this project
# https://www.tensorflow.org/js/tutorials/conversion/pretrained_model
# https://stackoverflow.com/questions/78216010/load-a-pretrained-model-in-tensorflowhttps://stackoverflow.com/questions/78216010/load-a-pretrained-model-in-tensorflow
# https://www.youtube.com/watch?v=dGuY1ytu1zs
# https://www.youtube.com/watch?v=idus3KO6Wic
# https://www.youtube.com/watch?v=iTlj3gMYzw8
# https://medium.com/@avinashmachinelearninginfo/sequences-and-padding-methods-in-nlp-in-tensorflow-ii-e5fabc06869b
# https://www.tensorflow.org/guide/keras/understanding_masking_and_padding
# https://github.com/gunthercox/ChatterBot/issues/1204
# https://www.youtube.com/watch?v=9L77QExPmI0
# https://docs.python.org/3/howto/logging.html
# https://www.geeksforgeeks.org/logging-in-python/
# https://www.datacamp.com/tutorial/pickle-python-tutorial
# https://www.geeksforgeeks.org/how-to-use-pickle-to-save-and-load-variables-in-python/


