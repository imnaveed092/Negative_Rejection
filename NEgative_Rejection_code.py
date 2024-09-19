from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from sklearn.metrics import precision_score, recall_score
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseEvaluator:
    """
    Evaluates the quality of generated responses against ground truth responses.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.5):
        """
        Initializes the evaluator with empty lists for true and generated responses.
        
        Args:
            model_name (str): The name of the Sentence Transformer model to use.
            similarity_threshold (float): The threshold for considering a word's meaning as not captured.
        """
        self.true_responses = []  # Ground truth responses
        self.generated_responses = []  # Generated responses
        self.model = SentenceTransformer(model_name)  # Load sentence transformer model
        self.similarity_threshold = similarity_threshold  # Similarity threshold for negative rejection

    def add_entry(self, true_response, generated_response):
        """
        Adds a new entry to the evaluator.

        Args:
            true_response (str): Ground truth response.
            generated_response (str): Generated response.
        """
        self.true_responses.append(true_response)
        self.generated_responses.append(generated_response)

    def tokenize_text(self, text):
        """
        Tokenizes text into sentences and then into words.

        Args:
            text (str): Text to tokenize.

        Returns:
            list: List of tokens (words).
        """
        sentences = sent_tokenize(text)
        tokens = [word_tokenize(sentence.lower()) for sentence in sentences]
        return [token for sublist in tokens for token in sublist]

    def calculate_negative_rejection_rate(self):
        """
        Calculates the negative rejection rate of the generated responses based on semantic similarity.

        Returns:
            float: Negative rejection rate.
        """
        total_words = 0
        total_not_captured = 0

        for true_response, generated_response in zip(self.true_responses, self.generated_responses):
            true_tokens = self.tokenize_text(true_response)
            gen_tokens = self.tokenize_text(generated_response)

            true_embeddings = self.model.encode(true_tokens)
            gen_embeddings = self.model.encode(gen_tokens)

            for true_embedding in true_embeddings:
                # Calculate similarities to all generated tokens
                similarities = cosine_similarity([true_embedding], gen_embeddings)[0]
                
                # Check if the highest similarity is below the threshold
                if max(similarities) < self.similarity_threshold:
                    total_not_captured += 1

            total_words += len(true_tokens)

        return total_not_captured / total_words if total_words > 0 else 0

    def calculate_precision(self):
            """
            Calculates precision based on the meaning similarity of tokens.

            Returns:
                float: Precision score.
            """
            true_embeddings = self.model.encode(self.true_responses)
            generated_embeddings = self.model.encode(self.generated_responses)

            y_true = []
            y_pred = []

            for true_emb, gen_emb in zip(true_embeddings, generated_embeddings):
                similarities = cosine_similarity([true_emb], [gen_emb])[0][0]
                # Consider it a positive prediction if similarity is above the threshold
                y_true.append(1)
                y_pred.append(1 if similarities >= self.similarity_threshold else 0)

            return precision_score(y_true, y_pred, zero_division=1)

    def calculate_recall(self):
            """
            Calculates recall based on the meaning similarity of tokens.

            Returns:
                float: Recall score.
            """
            true_embeddings = self.model.encode(self.true_responses)
            generated_embeddings = self.model.encode(self.generated_responses)

            y_true = []
            y_pred = []

            for true_emb, gen_emb in zip(true_embeddings, generated_embeddings):
                similarities = cosine_similarity([true_emb], [gen_emb])[0][0]
                # True response is expected, so we append 1 to y_true
                y_true.append(1)
                y_pred.append(1 if similarities >= self.similarity_threshold else 0)

            return recall_score(y_true, y_pred, zero_division=1)

    def calculate_meaning_similarity(self):
        true_embeddings = self.model.encode(self.true_responses)
        generated_embeddings = self.model.encode(self.generated_responses)
        
        similarities = [cosine_similarity([true], [gen])[0][0] for true, gen in zip(true_embeddings, generated_embeddings)]
        return np.mean(similarities)

    def flatten_responses(self):
        all_tokens = set([token for sublist in self.true_responses for token in sublist] +
                         [token for sublist in self.generated_responses for token in sublist])
        
        y_true = []
        y_pred = []

        for token in all_tokens:
            true_count = sum(token in true_set for true_set in self.true_responses)
            gen_count = sum(token in gen_set for gen_set in self.generated_responses)

            y_true.extend([1] * true_count + [0] * (len(self.true_responses) - true_count))
            y_pred.extend([1] * gen_count + [0] * (len(self.generated_responses) - gen_count))

        return y_true, y_pred
true_response="The meeting was postponed due to unforeseen circumstances."
gen_response="The meeting was delayed because of unexpected events."
# gen_response="I don't have any response on this" 

response_evaluator = ResponseEvaluator()
    
    # Add the current pair to the response evaluator
response_evaluator.add_entry(true_response, gen_response)

    # Calculate metrics

negative_rejection_rate = response_evaluator.calculate_negative_rejection_rate()
precision = response_evaluator.calculate_precision()
recall = response_evaluator.calculate_recall()
    # Print the metrics for the current pair
logger.info(f"Generated Response: {gen_response}")
logger.info(f"Ground Truth: {true_response}")
# logger.info(f"Precision: {precision:.2f}")
# logger.info(f"Recall: {recall:.2f}")
logger.info(f"Negative Rejection Rate: {negative_rejection_rate:.3f}")