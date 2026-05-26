import pytest

from services.kuzu_compat import ensure_kuzu_database_attribute


def test_ensure_kuzu_database_attribute_sets_missing_attribute():
    class Driver:
        pass

    driver = Driver()
    ensure_kuzu_database_attribute(driver, 'data/kuzu')

    assert driver._database == 'data/kuzu'


def test_ensure_kuzu_database_attribute_preserves_existing_attribute():
    class Driver:
        _database = 'existing'

    driver = Driver()
    ensure_kuzu_database_attribute(driver, 'data/kuzu')

    assert driver._database == 'existing'


def test_ensure_kuzu_database_attribute_handles_released_kuzu_driver(tmp_path):
    pytest.importorskip('kuzu')

    from graphiti_core.driver.kuzu_driver import KuzuDriver

    db_path = tmp_path / 'kuzu'
    driver = KuzuDriver(db=str(db_path))

    ensure_kuzu_database_attribute(driver, str(db_path))

    assert driver._database == str(db_path)
