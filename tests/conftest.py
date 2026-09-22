import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings


@pytest.fixture
def db():
    engine = create_engine(settings.database_url)

    connection = engine.connect()
    transaction = connection.begin()

    session = Session(
        bind=connection,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()