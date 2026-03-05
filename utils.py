
def split_text(text: str, max_chars: int = 5000):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + max_chars])
        start += max_chars
    return chunks

