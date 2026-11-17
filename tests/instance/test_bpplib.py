from pathlib import Path

import pytest

from bin_packing.instance.bpplib import read_instance, write_instance
from bin_packing.instance.instance import Instance


def test_written_instance_reads_back_identical(tmp_path: Path) -> None:
    original = Instance("toy", [6, 5, 4, 3, 2], 10)
    path = write_instance(original, tmp_path / "nested")
    assert path.read_text() == "5\n10\n6\n5\n4\n3\n2\n"
    restored = read_instance(path)
    assert restored.get_name() == "toy"
    assert restored.get_item_sizes() == original.get_item_sizes()
    assert restored.get_capacity() == original.get_capacity()


@pytest.mark.parametrize(
    ("content", "message"),
    [("3\n10\n1\n2\n", "declares 3 items but lists 2"), ("2\nten\n1\n1\n", "integers only"), ("5\n", "capacity")],
)
def test_reader_rejects_malformed_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "broken.txt"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_instance(path)
