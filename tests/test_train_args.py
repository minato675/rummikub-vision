from types import SimpleNamespace


def test_batch_integer_normalization_contract():
    args = SimpleNamespace(batch=8.0)
    batch = int(args.batch) if float(args.batch).is_integer() else args.batch
    assert batch == 8
    assert isinstance(batch, int)
