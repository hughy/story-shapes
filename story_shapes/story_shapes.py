#!/usr/bin/env python
import argparse
from collections import deque
from typing import Iterator
from typing import List

from transformers import pipeline

from story_shapes.plot import plot_story_shape

START_OF_STORY = "*** START OF THIS PROJECT GUTENBERG EBOOK"
END_OF_STORY = "*** END OF THIS PROJECT GUTENBERG EBOOK"


def main(story_path: str, story_title: str, shape_path: str) -> None:
    """Plots the shape of a story.

    Uses a pretrained sentiment analysis pipeline to compute the sentiment of
    segments of the story. The shape of the story is a plot of a rolling
    average of the story sentiment over a fixed-size window of segments.
    """
    sentiment_analyzer = pipeline("sentiment-analysis")
    tokenizer = sentiment_analyzer.tokenizer

    story_segments = iter_segments(story_path, tokenizer, 450)
    story_sentiments = iter_sentiments(sentiment_analyzer, story_segments)
    sentiment_averages = get_rolling_averages(story_sentiments, 10, 5)

    plot_story_shape(sentiment_averages, story_title, shape_path)


def get_rolling_averages(
    values: Iterator[float], window_length: int, window_stride: int = 1
) -> List[float]:
    """Computes a list of rolling averages of `values`.

    Each rolling average is the average over a window of `window_length`
    values from `values`. The value of `window_stride` gives the number of
    values that the window advances by in each step.
    """
    averages = []
    queue = deque()
    value_count = 0
    for value in values:
        value_count += 1
        queue.append(value)
        if len(queue) == window_length:
            averages.append(sum(queue) / window_length)
            for _ in range(window_stride):
                queue.popleft()

    # add any remaining partial window average
    expected_windows = (value_count - window_length) / window_stride + 1
    if len(averages) < expected_windows:
        averages.append(sum(queue) / len(queue))

    return averages


def iter_sentiments(sentiment_analyzer, segments: Iterator[str]) -> Iterator[float]:
    """Generates the sentiment score for each segment in `segments`.
    """
    for segment in segments:
        segment_sentiment = sentiment_analyzer(segment)[0]
        if segment_sentiment["label"] == "POSITIVE":
            yield segment_sentiment["score"]
        else:
            yield -segment_sentiment["score"]


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
    # subtract two for start and end tokens
    return len(tokenized_text["input_ids"]) - 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a graph of the shape of a story a la Kurt Vonnegut."
    )
    parser.add_argument(
        "--story-path", type=str, help="Filepath to read the story from."
    )
    parser.add_argument(
        "--title", type=str, help="The title of the story",
    )
    parser.add_argument(
        "--shape-path", type=str, help="Filepath to write the shape graph to.",
    )
    args = parser.parse_args()
    main(args.story_path, args.title, args.shape_path)
