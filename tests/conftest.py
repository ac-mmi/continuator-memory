import pytest

from continuator_memory.store import Store
from tests.helpers import HashingEmbedder


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()
