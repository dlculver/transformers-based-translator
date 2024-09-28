from tokenizers import ByteLevelBPETokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BLEUEvaluator:
    def __init__(self, tokenizer, pad_token_id, bos_token_id, eos_token_id):
        """
        Initialize the BLEU evaluator.
        
        Args:
            tokenizer: The tokenizer used for decoding tokenized sentences.
            pad_token_id: ID of the padding token.
            bos_token_id: ID of the beginning-of-sentence token.
            eos_token_id: ID of the end-of-sentence token.
        """
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.smoothing_function = SmoothingFunction().method1

    def remove_special_tokens(self, token_ids):
        """
        Remove special tokens like padding, BOS, and EOS from a list of token IDs.
        
        Args:
            token_ids: List of token IDs (integers).
        
        Returns:
            A list of token IDs with special tokens removed.
        """
        # Filter out special tokens
        return [id_ for id_ in token_ids if id_ not in {self.pad_token_id, self.bos_token_id, self.eos_token_id}]

    def decode_and_clean(self, token_ids):
        """
        Decode a list of token IDs into a sentence and remove special tokens.
        
        Args:
            token_ids: List of token IDs (integers).
        
        Returns:
            Decoded sentence (string) with special tokens removed.
        """
        # Remove special tokens first
        cleaned_ids = self.remove_special_tokens(token_ids)
        # Decode into a sentence
        return self.tokenizer.decode(cleaned_ids)

    def evaluate_batch_bleu(self, predicted_batch, reference_batch):
        """
        Evaluate the average BLEU score for a batch of predicted and reference sentences.
        
        Args:
            predicted_batch: List of lists, where each sublist is a predicted token IDs (e.g., [pred1, pred2, ...]).
            reference_batch: List of lists, where each sublist is a reference token IDs (e.g., [ref1, ref2, ...]).
        
        Returns:
            Average BLEU score for the batch.
        """
        total_bleu_score = 0.0
        num_sentences = len(predicted_batch)

        for predicted_ids, reference_ids in zip(predicted_batch, reference_batch):
            # Decode and clean both predicted and reference sentences
            predicted_sentence = self.decode_and_clean(predicted_ids)
            reference_sentence = self.decode_and_clean(reference_ids)

            # Tokenize the sentences into lists of words (BLEU expects tokenized sentences)
            predicted_tokens = predicted_sentence.split()
            reference_tokens = [reference_sentence.split()]  # List of lists for references

            # Compute BLEU score for the current sentence pair
            bleu_score = sentence_bleu(reference_tokens, predicted_tokens, smoothing_function=self.smoothing_function)
            total_bleu_score += bleu_score

        # Return the average BLEU score for the batch
        return total_bleu_score / num_sentences
