"""
Tests for Neo4j Schema Initialization (SRS §6.1.2, BUILD_GUIDE Task 2.2).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from graph_backend.build import (
    get_driver,
    parse_cypher_statements,
    run_schema_init,
)


class TestSchemaCypherParsing:
    """Tests for schema.cypher file parsing."""

    def test_schema_file_exists_and_contains_required_constraints(self):
        schema_path = Path(__file__).parent.parent / "graph_backend" / "schema.cypher"
        assert schema_path.exists(), "graph_backend/schema.cypher must exist"

        content = schema_path.read_text(encoding="utf-8")
        statements = parse_cypher_statements(content)

        assert len(statements) == 4, f"Expected 4 constraint statements, got {len(statements)}"

        stmts_text = " ".join(statements)
        assert "p.patient_id IS UNIQUE" in stmts_text
        assert "d.canonical_name IS UNIQUE" in stmts_text
        assert "m.canonical_name IS UNIQUE" in stmts_text
        assert "lt.canonical_name IS UNIQUE" in stmts_text

    def test_parse_cypher_statements_strips_comments(self):
        raw_cypher = """
        // Single line comment 1
        -- Single line comment 2
        /* Block comment
           across multiple lines */
        CREATE CONSTRAINT test1 IF NOT EXISTS FOR (a:A) REQUIRE a.id IS UNIQUE; // trailing comment
        CREATE CONSTRAINT test2 IF NOT EXISTS FOR (b:B) REQUIRE b.id IS UNIQUE;
        """
        statements = parse_cypher_statements(raw_cypher)
        assert len(statements) == 2
        assert "test1" in statements[0]
        assert "test2" in statements[1]
        assert "//" not in statements[0]
        assert "/*" not in statements[0]


class TestSchemaInitExecution:
    """Tests for run_schema_init execution with mocked Neo4j driver."""

    def test_run_schema_init_with_mock_driver(self, tmp_path):
        dummy_schema = tmp_path / "test_schema.cypher"
        dummy_schema.write_text(
            "CREATE CONSTRAINT c1 IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE;\n"
            "CREATE CONSTRAINT c2 IF NOT EXISTS FOR (d:Diagnosis) REQUIRE d.name IS UNIQUE;\n",
            encoding="utf-8",
        )

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        executed = run_schema_init(driver=mock_driver, schema_path=dummy_schema)

        assert len(executed) == 2
        assert mock_session.run.call_count == 2

    def test_run_schema_init_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            run_schema_init(driver=MagicMock(), schema_path="non_existent_file.cypher")


class TestGetDriver:
    """Tests for get_driver connection setup and validation."""

    def test_missing_credentials_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "")
        monkeypatch.setenv("NEO4J_PASSWORD", "")

        with pytest.raises(ValueError, match="NEO4J_URI or NEO4J_PASSWORD is not set"):
            get_driver()

    @patch("neo4j.GraphDatabase.driver")
    def test_get_driver_success(self, mock_driver_factory, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "neo4j+s://test.databases.neo4j.io")
        monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")

        mock_driver = MagicMock()
        mock_driver_factory.return_value = mock_driver

        driver = get_driver()
        assert driver == mock_driver
        mock_driver.verify_connectivity.assert_called_once()
