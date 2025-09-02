import re
from typing import Iterable, Iterator
import pickle


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """Construct a tokenizer from a given vocabulary, merges and special tokens

        Args:
            vocab (dict[int,bytes]): vocabulary
            merges (list[tuple[bytes, bytes]]): merges in the order of creation
            special_tokens (list[str], optional): list of special tokens. Defaults to None.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Construct and return a Tokenizer from a serialized vocabulary and a list of merges.

        Args:
            vocab_filepath (str)
            merges_filepath (str): _description_
            special_tokens (list[str], optional): _description_. Defaults to None.
        """
        # load the pickle object from from local file
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    
    def encode(self, text: str)-> list[int]:
        """Encode an input text into a sequence of token IDs"""        
        # Step0: remove special tokens to split into seperate chunks
        special_tokens = self.special_tokens
        chunks = [text]   # init as a list of text itself
        import re
        if special_tokens:  # special tokens not none
            escaped_tokens = [re.escape(token) for token in special_tokens]
            special_tokens_pattern = '(' + '|'.join(escaped_tokens) + ')'  # use special pattern to parse the text
            chunks = re.split(special_tokens_pattern, text)   # split with special tokens as delimeter
        # chunks are text seperated by special tokens
        # E.g. ['What is the', <sp1>, ' are you k', <sp2>, 'idding me? '], each element in the list is a chunk
                
        # Step1: pretokenization for each chunk
        pretokenized_chunks = []
        for chunk in chunks:
            # pretokenization
            import regex as re
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pretoken_list = []  # the converted pretoken list for this doc chunk
            if chunk in special_tokens:  # check whether the document is a standalone special tokens 
                pretoken = chunk
                pretoken_list.append(pretoken)
            else:
                # parse the doc into list of pretokens
                pretokens = re.finditer(PAT, chunk)
                for match in pretokens:
                    pretoken = match.group()
                    pretoken_list.append(pretoken)

            pretokenized_chunks.append(pretoken_list)
        # pretokenized_chunks is a list of list of pretokens. 
        # [['What', ' is', ' the'], ['<hi>'], [' are', ' you', ' k'], ['|<endoftext>|'], ['idding', ' me', '?', ' ']]
        
        # Step2: match the merges
        all_id_sequence = []
        vocab = self.vocab
        merges = self.merges
        for chunk in pretokenized_chunks:            
            chunk_id_sequence = []
            for pretoken in chunk:
                byte_seq = pretoken.encode('utf-8')  # pretoken -> b'pretoken'
                token_id_sequence_for_pretoken = []
                
                # Step 2.0: check for special tokens
                if pretoken in special_tokens:
                    assert len(chunk) == 1 # there should be only this special token in this chunk
                    # directly to token id
                    
                    # TODO: given a byte sequence of a token, find its token id can be wrapped into a function
                    # find the token id from the vocab
                    token_id = None
                    for key, value in vocab.items():
                        if value == byte_seq:
                            token_id = key
                            break
                        
                    # add the token id to the sequence
                    if not token_id:
                        raise Exception("Could not find special token's token id in the vocabulary")
                    chunk_id_sequence.append([token_id])
                    
                    # Move to the next chunk
                    break
                    
                # Step2.1: convert into utf-8 encoding 
                byte_list = []
                for i in range(len(byte_seq)):
                    byte_list.append(bytes([byte_seq[i]]))
                # print(f"the byte list for pretoken {pretoken} is: {byte_list}")
                        
                # Step2.2: merge byte pair according to merges, get a list of token for each pretoken
                can_merge = True
                
                # Step2.2.1: get the list of all byte pairs for the pretoken
                byte_pair_list = list()   
                for i in range(len(byte_list) - 1):
                    byte_pair_list.append((byte_list[i], byte_list[i+1]))
                    
                # Iterative merging
                while(can_merge):
                    byte_pair_set = set(byte_pair_list)
                        
                    # Step2.2.2: check whether existing pairs merge in merges, if so, find the first one. 
                    can_merge = False
                    for item in merges:  
                        if item in byte_pair_set:  # the first element is the one
                            can_merge = True
                            pair_to_merge = item
                            break
                    
                    # Step2.2.3: merge the pair in the list
                    if can_merge:
                        merged_bytes = pair_to_merge[0] + pair_to_merge[1]
                        new_byte_list = []
                        cursor = 0
                        while (cursor < len(byte_list)):
                            # if at end of list 
                            if cursor == len(byte_list) - 1:
                                new_byte_list.append(byte_list[cursor])
                                break
                            else: # can form a byte pair
                                the_pair = (byte_list[cursor], byte_list[cursor + 1])
                                if the_pair == pair_to_merge:
                                    new_byte_list.append(merged_bytes)
                                    cursor += 2
                                else:
                                    new_byte_list.append(byte_list[cursor])
                                    cursor += 1
                        byte_list = new_byte_list
                        
                    # Update the byte pair list after merging complted
                    byte_pair_list = list()  
                    for i in range(len(byte_list) - 1):
                        byte_pair_list.append((byte_list[i], byte_list[i+1]))

                                    
                # Step 2.3: convert each byte into token id
                for byte_seq in byte_list:
                    token_id = None
                    for key, value in vocab.items():
                        if value == byte_seq:
                            token_id = key
                            break
                        
                    # add the token id to the sequence
                    if not token_id:
                        raise Exception(f"Could not find token id for byteseq {byte_seq} in the vocabulary")
                    token_id_sequence_for_pretoken.append(token_id)
                chunk_id_sequence.append(token_id_sequence_for_pretoken)
                
            # append the chunk's token id into all tokens'
            all_id_sequence.append(chunk_id_sequence)
            
        flat_id_sequence = []
        for chunk in all_id_sequence:
            for pretoken in chunk:
                for token in pretoken:
                    flat_id_sequence.append(token)
        return flat_id_sequence
    
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

    
    