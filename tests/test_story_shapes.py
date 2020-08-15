from transformers import AutoTokenizer

from story_shapes import __version__
from story_shapes.story_shapes import END_OF_STORY
from story_shapes.story_shapes import START_OF_STORY
from story_shapes.story_shapes import iter_segments
from story_shapes.story_shapes import iter_windows


def test_version():
    assert __version__ == '0.1.0'


def test_iter_segments(tmpdir):
    # iter_segments does not break lines, so use one word per line for testing
    pre_story = "The\nquick\nbrown\nfox\nran."
    test_story = "The\nquick\nbrown\nfox\njumped\nover\nthe\nlazy\ndog."
    post_story = "The\nlazy\ndog\nbarked."

    test_text = "\n".join([pre_story, START_OF_STORY, test_story, END_OF_STORY, post_story])
    test_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    test_path = tmpdir / "test.txt"
    with open(test_path, "w+") as f:
        f.write(test_text)

    segments = list(iter_segments(test_path, test_tokenizer, 1))
    print(segments)
    assert len(segments) == 9  # nine lines in test_story
    segments = list(iter_segments(test_path, test_tokenizer, 4))
    print(segments)
    assert len(segments) == 3


def test_iter_windows():
    test_segments = range(10)

    windows_list = list(iter_windows(test_segments, 4))
    assert len(windows_list) == 7
    assert all(len(window) == 4 for window in windows_list)
