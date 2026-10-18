"""
Tests for tolerant parsing of local-model output.

The local GGUF model ignores "reply with JSON only": it emits a bare `thought`
channel marker, several hundred words of reasoning, and JSON that is frequently
truncated mid-object. The previous strip-and-json.loads pipeline raised on every
such reply and dropped the whole chunk, which is how a rebuild on the local model
produced a graph containing no clinical facts.
"""

import pytest

from llm_client import (
    parse_json_list,
    parse_json_object,
    strip_model_preamble,
    _salvage_objects,
)


class TestStripModelPreamble:
    @pytest.mark.parametrize("raw,expected", [
        ('thought\n[{"a": 1}]', '[{"a": 1}]'),
        ('<thought>\n[{"a": 1}]', '[{"a": 1}]'),
        ('model\n[{"a": 1}]', '[{"a": 1}]'),
        ('```json\n[{"a": 1}]\n```', '[{"a": 1}]'),
        ('<thought>reasoning here</thought>\n[{"a": 1}]', '[{"a": 1}]'),
        ('[{"a": 1}]', '[{"a": 1}]'),
    ])
    def test_removes_channels_and_fences(self, raw, expected):
        assert strip_model_preamble(raw) == expected

    def test_handles_empty(self):
        assert strip_model_preamble("") == ""
        assert strip_model_preamble(None) == ""


class TestSalvageObjects:
    def test_recovers_complete_objects_from_a_truncated_array(self):
        truncated = '[{"a": 1}, {"a": 2}, {"a": "unterminat'
        assert _salvage_objects(truncated) == [{"a": 1}, {"a": 2}]

    def test_braces_inside_strings_do_not_confuse_it(self):
        text = '[{"note": "value with { and } inside"}, {"b": 2}]'
        assert _salvage_objects(text) == [{"note": "value with { and } inside"}, {"b": 2}]

    def test_escaped_quote_inside_a_string(self):
        text = '[{"note": "he said \\"hi\\" loudly"}, {"b": 2}]'
        assert _salvage_objects(text) == [{"note": 'he said "hi" loudly'}, {"b": 2}]

    def test_nested_objects_are_kept_whole(self):
        text = '[{"a": {"inner": 1}}, {"b": 2}]'
        assert _salvage_objects(text) == [{"a": {"inner": 1}}, {"b": 2}]

    def test_nothing_to_salvage(self):
        assert _salvage_objects("no json here at all") == []


class TestParseJsonList:
    def test_plain_array(self):
        assert parse_json_list('[{"a": 1}]') == [{"a": 1}]

    def test_single_object_becomes_a_list(self):
        assert parse_json_list('{"a": 1}') == [{"a": 1}]

    def test_reasoning_preamble_then_array(self):
        reply = (
            "thought\n"
            "The user wants triples. Let me think about this.\n"
            "**Entities:** Diagnosis: cancer\n\n"
            '[{"relation": "STATES_DIAGNOSIS", "object_name": "Breast Cancer"}]'
        )
        assert parse_json_list(reply) == [
            {"relation": "STATES_DIAGNOSIS", "object_name": "Breast Cancer"}
        ]

    def test_truncated_array_after_reasoning_keeps_complete_facts(self):
        reply = (
            "thought\nLong reasoning. Relationships: (Report)-[:STATES_DIAGNOSIS]->(Diagnosis)\n\n"
            '[\n  {"relation": "STATES_DIAGNOSIS", "object_name": "Ascites"},\n'
            '  {"relation": "UNDERWENT", "object_name": "CECT"},\n'
            '  {"relation": "HAS_RESU'
        )
        got = parse_json_list(reply)
        assert [o["object_name"] for o in got] == ["Ascites", "CECT"]

    def test_prose_brackets_and_odd_quotes_do_not_derail_it(self):
        """Reasoning carries Cypher-style '[' and an odd number of quotes."""
        reply = (
            'thought\nI will use (Report)-[:STATES_DIAGNOSIS]->(Diagnosis) and note "Cycle 3\n'
            'and an unclosed quote makes brace scanning unreliable.\n\n'
            '[{"relation": "STATES_DIAGNOSIS", "object_name": "Metastatic Breast Cancer"}]'
        )
        assert parse_json_list(reply) == [
            {"relation": "STATES_DIAGNOSIS", "object_name": "Metastatic Breast Cancer"}
        ]

    def test_trailing_prose_after_the_array(self):
        reply = '[{"a": 1}]\n\nI hope this helps!'
        assert parse_json_list(reply) == [{"a": 1}]

    def test_empty_and_unparseable_give_an_empty_list(self):
        assert parse_json_list("") == []
        assert parse_json_list("I could not find anything.") == []
        assert parse_json_list("[]") == []


class TestParseJsonObject:
    def test_plain_object(self):
        assert parse_json_object('{"canonical": "Hemoglobin", "type": "LabTest"}') == {
            "canonical": "Hemoglobin", "type": "LabTest"
        }

    def test_object_after_reasoning(self):
        reply = 'thought\nThis looks like a lab test.\n\n{"canonical": "Hemoglobin", "type": "LabTest"}'
        assert parse_json_object(reply)["canonical"] == "Hemoglobin"

    def test_literal_null_means_no_mapping(self):
        assert parse_json_object("null") is None
        assert parse_json_object("  NULL  ") is None

    def test_empty_or_unparseable(self):
        assert parse_json_object("") is None
        assert parse_json_object("no json") is None

    def test_an_array_is_not_an_object(self):
        assert parse_json_object('[1, 2, 3]') is None
