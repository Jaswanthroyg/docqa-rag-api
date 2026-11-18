import re


def split_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most `size` characters.

    Paragraph and sentence boundaries are preferred over arbitrary cuts, and each
    chunk starts with the tail of the previous one (`overlap`) so answers that
    straddle a boundary are still retrievable.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        units.extend([para] if len(para) <= size else _split_long(para, size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= size:
            current += "\n\n" + unit
        else:
            chunks.append(current)
            tail = _tail(current, overlap)
            current = f"{tail}\n\n{unit}" if tail and len(tail) + 2 + len(unit) <= size else unit
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= size:
            pieces.append(sentence)
        else:  # no sentence breaks: cut on words, then hard-cut oversized tokens
            pieces.extend(_hard_cut(sentence, size))
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= size:
            current += " " + piece
        else:
            packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return packed


def _hard_cut(text: str, size: int) -> list[str]:
    out: list[str] = []
    current = ""
    for word in text.split(" "):
        while len(word) > size:
            if current:
                out.append(current)
                current = ""
            out.append(word[:size])
            word = word[size:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= size:
            current += " " + word
        else:
            out.append(current)
            current = word
    if current:
        out.append(current)
    return out


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail
