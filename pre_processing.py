import os
import json

from configs import raw_data, chunk_len
from tokenizers.pre_tokenizers import BertPreTokenizer


def tokenize(text: str, tokenizer: BertPreTokenizer) -> list[tuple]:
    """
    Use a word-level tokenizer to split text into tokens and keep track of spans.

    :text: input text to tokenize
    
    return: list of tuples in the form (token, start_span, end_span)
    """
    out = []
    for word, (start, stop) in tokenizer.pre_tokenize_str(text):
        out.append((word, start, stop))

    return out

def assign_labels(tokens: list, annotations: list) -> list[str]:
    """
    Match the tokens with the correct labels based on the annotations. The annotations contain the "source of truth", as for 
    whether a token is PII or not.
    
    "DIRECT" or "QUASI" in the field identifier_type of the annotation means that it is PII. Anything else means that it is not PII. 

    BERT requires data to be in the IOB format. Label key below:
    B = beginning of PII span
    I = inside of a PII span
    O = not PII

    :tokens: tokens to label
    :annotations: source of truth to match with tokens

    return: list of labels corresponding to the tokens in IOB format
    """
    # initialize a list of O tokens, then replace the relevant ones with B/I
    labels = ["O"] * len(tokens)

    # find the spans that are PII 
    spans = []
    for a in annotations:
        if a["identifier_type"] in ["DIRECT", "QUASI"]:
            spans.append((a["start_offset"], a["end_offset"]))

    # sort in "chronological order"
    # aka from lowest to highest based on the start_offset
    spans.sort()

    # loop through spans and mark B and I
    for span in spans:
        span_start = span[0]
        span_end = span[1]

        # find out how many tokens are in this span
        # note: this also includes when tokens are partially in the span
        tok_positions = []
        for i in range(len(tokens)):
            tok_start = tokens[i][1]
            tok_end = tokens[i][2]
            if tok_start < span_end and tok_end > span_start:
                tok_positions.append(i) # append just index of the token

        # skip if there are no overlaps found
        if len(tok_positions) == 0:
            continue

        # label B and I 
        first = True
        for i in tok_positions:
            if labels[i] == "O":
                if first:
                    labels[i] = "B"
                else:
                    labels[i] = "I"
            first = False
    
    return labels

def chunk(tokens: list, labels: list, chunk_len: int) -> list[tuple]:
    """
    Break document into chunks of max chunk_len tokens. 
    If PII spans the chunk boundary, it is cut before (last chunk might be smaller than chunk_len). 

    :tokens: the tokenized text in the current case
    :labels: labels associated with each token
    :chunk_len: the size of chunks to split the tokens into

    return: list of chunks of (tokens,labels)
    """
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = start_idx + chunk_len

        if end_idx >= len(tokens):
            # if the remaining tokens are fewer than chunk_len, 
            # then take the rest of the tokens as the last chunk
            end_idx = len(tokens)

        # make sure that the chunk will not split a PII span
        # labels[end_idx] is the first label of the next chunk
        elif labels[end_idx] == "I":
            # iterate to the left in the tokens until the beginning of the PII span is found
            # then split there instead
            while True:
                if end_idx <= start_idx + 1:
                    break
                if labels[end_idx] == "B":
                    break
                # shift once to the left and continue
                end_idx = end_idx -1

            # rare case: if the PII spans the whole chunk, just use the chunk_len as is.
            # let the PII span be split since there is nothing to do about it
            if end_idx == start_idx + 1:
                end_idx = start_idx + chunk_len

        curr_tokens = tokens[start_idx:end_idx]
        curr_labels = labels[start_idx:end_idx]

        chunks.append((curr_tokens, curr_labels))
                
        # set idx for the next chunk
        start_idx = end_idx

    return chunks
    
    

def convert_to_iob(corpus: list) -> list:
    """
    Convert the data into the IOB format for BERT.

    :corpus: list of unprocessed corpus data 

    return: tokenized, chunked and formatted corpus data with tokens, labels and spans. 
    """
    iob_data = []
    bert_tok = BertPreTokenizer()
    
    # loop through corpus and extract the useful info (first annotation and text)
    for case in corpus:
        annotations = list(case["annotations"].values())
        first_annotation = annotations[0]["entity_mentions"]
        tokens = tokenize(case["text"], bert_tok)
        labels = assign_labels(tokens, first_annotation)
        
        # chunk the text at len chunk_len, and save to iob_data dict
        chunks = chunk(tokens, labels, chunk_len)
        i = 0
        for curr_toks, curr_labels in chunks:
            words = []
            spans = []
            for word, start, stop in curr_toks:
                words.append(word)
                spans.append((start, stop))

            iob_data.append({
                "chunk_idx": i,
                "tokens": words,
                "labels": curr_labels,
                "spans": spans
            })
            
            i += 1
    
    return iob_data


if __name__ == "__main__":

    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("text-anonymization-benchmark"):
        raise FileNotFoundError("Clone the text-anonymization-benchmark repo before proceeding.")

    # process data, then save it
    for data_split, path in raw_data.items():
        # read raw data from current split
        with open(path, encoding="utf-8") as f:
            corpus = json.load(f)
        
        # tokenize the text and convert labels to IOB format
        iob_data = convert_to_iob(corpus)
    
        # save processed data
        with open(f"data/{data_split}.json", "w", encoding="utf-8") as f:
            json.dump(iob_data, f, ensure_ascii=False) # make sure all language chars are preserved
            print(f"saved data to data/{data_split}.json")