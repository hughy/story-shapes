#!/usr/bin/env python
from collections import deque
from copy import deepcopy
from typing import Iterator
from typing import List

from transformers import pipeline

from story_shapes.plot import plot_shape

START_OF_STORY = "*** START OF THIS PROJECT GUTENBERG EBOOK"
END_OF_STORY = "*** END OF THIS PROJECT GUTENBERG EBOOK"


def main(story_path: str) -> None:
    sentiment_analyzer = pipeline("sentiment-analysis")
    tokenizer = sentiment_analyzer.tokenizer

    segments = iter_segments(story_path, tokenizer, 50)

    sentiments = []
    for window in iter_windows(segments, 5):
        window_text = "\n".join(window)
        window_sentiment = sentiment_analyzer(window_text)[0]
        if window_sentiment["label"] == "POSITIVE":
            sentiments.append(window_sentiment["score"])
        else:
            sentiments.append(-window_sentiment["score"])

    plot_shape(sentiments)


def iter_windows(segments: Iterator[str], window_length: int) -> Iterator[List[str]]:
    """Yields a sliding window of `window_length` segments.
    """
    queue = deque()
    for segment in segments:
        queue.append(segment)
        if len(queue) == window_length:
            yield deepcopy(queue)
            queue.popleft()


def iter_segments(story_path: str, tokenizer, segment_length: int) -> Iterator[str]:
    """Breaks a story into segments of roughly equal length in tokens.

    Reads the story at `story_path` one line at a time and uses the given
    `tokenizer` to count the number of tokens in each line. Groups lines into
    a segment as long as the number of tokens is less than `segment_length`.

    Makes some attempt to exclude lines that are not part of the story by
    looking for lines that mark the start and end of a Project Gutenberg
    Ebook.
    """
    with open(story_path) as f:
        while line := f.readline():
            if line.startswith(START_OF_STORY):
                break

        segment = ""
        segment_tokens = 0
        while line := f.readline():
            if line.startswith(END_OF_STORY):
                break

            segment += line
            segment_tokens += get_token_count(tokenizer, line)
            if segment_tokens >= segment_length:
                yield segment
                segment = ""
                segment_tokens = 0

    # yield any remaining segment under segment_length
    if segment:
        yield segment


def get_token_count(tokenizer, text: str) -> int:
    tokenized_text = tokenizer(text)
    return len(tokenized_text["input_ids"]) - 2


if __name__ == "__main__":
    main("beowulf.txt")
