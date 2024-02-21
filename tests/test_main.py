from llamaindex_course.main import calculate


def test_calculate():
    """ Test the calculate function """
    assert calculate() == 4
