from typing import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """Construct a tokenizer from a given vocabulary, merges and special tokens

        Args:
            vocab (dict[int,bytes]): vocabulary
            merges (list[tuple[bytes, bytes]]): merges in the order of creation
            special_tokens (list[str], optional): list of special tokens. Defaults to None.
        """
        pass
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Construct and return a Tokenizer from a serialized vocabulary and a list of merges.

        Args:
            vocab_filepath (str)
            merges_filepath (str): _description_
            special_tokens (list[str], optional): _description_. Defaults to None.
        """
        pass
    
    def encode(self, text: str)-> list[int]:
        """Encode an input text into a sequence of token IDs"""
        pass
    
    def encode_iterable(self, iterable:Iterable[str])-> Iterator[int]:
        """
        Given an interable of strings, return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that cannot be directly loaded into memory. 
        """
        pass
    
    def decode(self, ids:list[int])-> str:
        """
        Decode a sequence of token IDs into text.
        """
        pass

    
    