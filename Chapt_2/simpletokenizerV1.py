import re

class SimpleTokenizerV1:
    def __init__(self, vocab):              
        self.str_to_int = vocab             #1 Stores the vocabulary as a class attribute for access in the encode and decode methods
        self.int_to_str = {i:s for s,i in vocab.items()}       #2 Creates an inverse vocabulary that maps token IDs back to the original text tokens

    def encode(self, text):             #3 Processes input text into token IDs
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):              #4 Converts token IDs back into text
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)             #5 Removes spaces before the specified punctuation
        return text
    
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know." Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)