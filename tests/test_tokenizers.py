from minigpt.tokenizers import ByteTokenizer, CharTokenizer


def test_char_tokenizer_round_trip():
    text = "hello world"
    tokenizer = CharTokenizer(text=text)

    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size == len(set(text))


def test_byte_tokenizer_round_trip_ascii():
    text = "hello world"
    tokenizer = ByteTokenizer()

    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size == 256


def test_byte_tokenizer_round_trip_unicode():
    text = "hello 世界"
    tokenizer = ByteTokenizer()

    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size == 256