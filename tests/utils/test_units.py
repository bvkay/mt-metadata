import pytest
from loguru import logger

from mt_metadata.common import units
from mt_metadata.common.units import (
    find_separator,
    get_unit_from_df,
    get_unit_object,
    parse_unit_string,
    Unit,
    UNITS_DF,
)


class TestUnit:
    """Test class for Unit functionality with fixtures and subtests."""

    @pytest.fixture
    def unit_data(self):
        """Fixture providing unit test data."""
        return {
            "name": "millivolt",
            "description": "Unit of electric potential",
            "symbol": "mV",
            "plot_label": "mV",
        }

    @pytest.fixture
    def sample_unit(self, unit_data):
        """Fixture providing a sample Unit instance."""
        return Unit(**unit_data)

    @pytest.fixture
    def unit_pairs(self):
        """Fixture providing unit pairs for combination tests."""
        return [
            {
                "unit1": Unit(
                    name="millivolt",
                    symbol="mV",
                    description="Millivolt",
                    plot_label="mV",
                ),
                "unit2": Unit(
                    name="nanotesla",
                    symbol="nT",
                    description="Nanotesla",
                    plot_label="nT",
                ),
                "separator": "/",
                "expected": {
                    "name": "millivolt per nanotesla",
                    "symbol": "mV/nT",
                    "description": "Millivolt per Nanotesla",
                    "plot_label": "mV/nT",
                },
            },
            {
                "unit1": Unit(
                    name="millivolt",
                    symbol="mV",
                    description="Millivolt",
                    plot_label="mV",
                ),
                "unit2": Unit(
                    name="nanotesla",
                    symbol="nT",
                    description="Nanotesla",
                    plot_label="nT",
                ),
                "separator": " ",
                "expected": {
                    "name": "millivolt nanotesla",
                    "symbol": "mV nT",
                    "description": "Millivolt Nanotesla",
                    "plot_label": "mV nT",
                },
            },
        ]

    def test_unit_initialization(self, subtests, unit_data):
        """Test the initialization of the Unit class."""
        unit = Unit(**unit_data)
        with subtests.test("name"):
            assert unit.name == unit_data["name"]
        with subtests.test("description"):
            assert unit.description == unit_data["description"]
        with subtests.test("symbol"):
            assert unit.symbol == unit_data["symbol"]
        with subtests.test("plot_label"):
            assert unit.plot_label == unit_data["plot_label"]

    def test_unit_to_dict(self, sample_unit, unit_data):
        """Test the to_dict method of the Unit class."""
        unit_dict = sample_unit.to_dict()
        assert unit_dict == unit_data

    def test_unit_from_dict(self, subtests, unit_data):
        """Test the from_dict method of the Unit class."""
        unit = Unit(name="", description="", symbol="", plot_label="")
        unit.from_dict(unit_data)

        with subtests.test("name"):
            assert unit.name == unit_data["name"]
        with subtests.test("description"):
            assert unit.description == unit_data["description"]
        with subtests.test("symbol"):
            assert unit.symbol == unit_data["symbol"]
        with subtests.test("plot_label"):
            assert unit.plot_label == unit_data["plot_label"]

    def test_unit_combine(self, subtests, unit_pairs):
        """Test the combine method of the Unit class."""
        for pair in unit_pairs:
            with subtests.test(separator=pair["separator"]):
                combined_unit = pair["unit1"].combine(
                    pair["unit2"], separator=pair["separator"]
                )
                expected = pair["expected"]

                with subtests.test("name", separator=pair["separator"]):
                    assert combined_unit.name == expected["name"]
                with subtests.test("symbol", separator=pair["separator"]):
                    assert combined_unit.symbol == expected["symbol"]
                with subtests.test("description", separator=pair["separator"]):
                    assert combined_unit.description == expected["description"]
                with subtests.test("plot_label", separator=pair["separator"]):
                    assert combined_unit.plot_label == expected["plot_label"]


class TestUtilityFunctions:
    """Test class for utility functions with fixtures and subtests."""

    @pytest.fixture
    def separator_test_cases(self):
        """Fixture providing test cases for find_separator function."""
        return [
            {"input": "mV nT/[km ohm]", "expected": " "},
            {"input": "mV/nT", "expected": "/"},
            {"input": "mV per nT", "expected": " per "},
            {"input": "mVnT", "expected": None},
        ]

    def test_find_separator(self, subtests, separator_test_cases):
        """Test the find_separator function."""
        for case in separator_test_cases:
            with subtests.test(input_string=case["input"]):
                assert find_separator(case["input"]) == case["expected"]

    @pytest.fixture
    def parse_unit_test_cases(self):
        """Fixture providing test cases for parse_unit_string function."""
        return [
            {
                "input": "mV nT/[km ohm]",
                "expected": [
                    {"name": "mV", "sep": ""},
                    {"name": "nT", "sep": " "},
                    {"name": "km", "sep": "/"},
                    {"name": "ohm", "sep": " "},
                ],
            },
            {
                "input": "mV/nT",
                "expected": [
                    {"name": "mV", "sep": ""},
                    {"name": "nT", "sep": "/"},
                ],
            },
            {
                "input": "mV per nT",
                "expected": [
                    {"name": "mV", "sep": ""},
                    {"name": "nT", "sep": " per "},
                ],
            },
        ]

    @pytest.fixture
    def unit_from_df_test_cases(self):
        """Fixture providing test cases for get_unit_from_df function."""
        return [
            {"input": "mV", "expected_name": "milliVolt", "expected_symbol": "mV"},
            {"input": "Hz", "expected_name": "hertz", "expected_symbol": "Hz"},
        ]

    @pytest.fixture
    def unit_object_test_cases(self):
        """Fixture providing test cases for get_unit_object function."""
        return [
            {"input": "mV", "expected_name": "milliVolt", "expected_symbol": "mV"},
            {
                "input": "mV/nT",
                "expected_name": "milliVolt per nanoTesla",
                "expected_symbol": "mV/nT",
            },
            {
                "input": "mV nT/[km ohm]",
                "expected_name": "milliVolt nanoTesla per kilometer Ohm",
                "expected_symbol": "mV nT/km \u03a9",
            },
        ]

    def test_parse_unit_string(self, subtests, parse_unit_test_cases):
        """Test the parse_unit_string function."""
        for case in parse_unit_test_cases:
            with subtests.test(input_string=case["input"]):
                assert parse_unit_string(case["input"]) == case["expected"]

    def test_parse_unit_string_empty_error(self):
        """Test parse_unit_string with empty string raises ValueError."""
        with pytest.raises(ValueError):
            parse_unit_string("")

    def test_get_unit_from_df(self, subtests, unit_from_df_test_cases):
        """Test the get_unit_from_df function."""
        for case in unit_from_df_test_cases:
            with subtests.test(unit_symbol=case["input"]):
                unit = get_unit_from_df(case["input"])
                assert isinstance(unit, Unit)
                assert unit.name == case["expected_name"]
                assert unit.symbol == case["expected_symbol"]

    def test_get_unit_from_df_errors(self, subtests):
        """Test get_unit_from_df error handling."""
        with subtests.test("KeyError on nonexistent unit"):
            with pytest.raises(KeyError):
                get_unit_from_df("nonexistent_unit", allow_none=False)

        with subtests.test("Unknown unit when allow_none=True"):
            unit = get_unit_from_df("nonexistent_unit", allow_none=True)
            assert unit.name == "unknown"
            assert unit.symbol == "unknown"

    def test_get_unit_object(self, subtests, unit_object_test_cases):
        """Test the get_unit_object function."""
        for case in unit_object_test_cases:
            with subtests.test(unit_string=case["input"]):
                unit = get_unit_object(case["input"])
                assert isinstance(unit, Unit)
                assert unit.name == case["expected_name"]
                assert unit.symbol == case["expected_symbol"]

    def test_get_unit_object_type_error(self):
        """Test get_unit_object with invalid input."""
        # Test with empty string which should handle gracefully
        with pytest.raises(ValueError):
            parse_unit_string("")


class _CountingTable:
    """Stands in for UNITS_DF and counts the lookups made on it."""

    def __init__(self, table):
        self.table = table
        self.lookups = 0

    def __getitem__(self, key):
        self.lookups += 1
        return self.table[key]


class TestGetUnitFromDfCache:
    """Rows found in UNITS_DF are cached; unknown units are not."""

    def test_second_lookup_skips_table(self, monkeypatch):
        table = _CountingTable(UNITS_DF)
        monkeypatch.setattr(units, "UNITS_DF", table)

        first = get_unit_from_df("mV")
        lookups = table.lookups
        second = get_unit_from_df("mV")

        assert lookups > 0
        assert table.lookups == lookups
        assert second.model_dump() == first.model_dump()

    @pytest.mark.parametrize(
        "value", ["mV", "milliVolt", "millivolt", "nT", "Hz", "M", "digital counts"]
    )
    def test_cached_equals_uncached(self, monkeypatch, value):
        get_unit_from_df(value)
        cached = get_unit_from_df(value)
        # replacing the table empties the cache, so this lookup searches the table
        monkeypatch.setattr(units, "UNITS_DF", UNITS_DF.copy())
        uncached = get_unit_from_df(value)

        assert cached.model_dump() == uncached.model_dump()

    def test_returned_unit_is_a_new_object(self):
        first = get_unit_from_df("mV")
        first.description = "changed"
        second = get_unit_from_df("mV")

        assert second is not first
        assert second.description != "changed"

    def test_replaced_table_is_used(self, monkeypatch):
        get_unit_from_df("mV")
        table = UNITS_DF.copy()
        table.loc[table["symbol"] == "mV", "description"] = "changed"
        monkeypatch.setattr(units, "UNITS_DF", table)

        assert get_unit_from_df("mV").description == "changed"

    def test_unknown_unit_warns_every_call(self):
        messages = []
        sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            units_out = [get_unit_from_df("not_a_unit") for _ in range(2)]
        finally:
            logger.remove(sink)

        assert [u.name for u in units_out] == ["unknown", "unknown"]
        assert len([m for m in messages if "'not_a_unit' not found" in m]) == 2

    def test_unknown_unit_raises_every_call(self):
        for _ in range(2):
            with pytest.raises(KeyError, match="not_a_unit"):
                get_unit_from_df("not_a_unit", allow_none=False)


class TestUnitsDataFrame:
    """Test class for UNITS_DF functionality."""

    def test_units_df_structure(self, subtests):
        """Test the UNITS_DF DataFrame structure."""
        with subtests.test("DataFrame not empty"):
            assert not UNITS_DF.empty

        required_columns = ["name", "symbol", "description", "plot_label"]
        for column in required_columns:
            with subtests.test(f"Has {column} column"):
                assert column in UNITS_DF.columns

    def test_units_df_known_unit(self, subtests):
        """Test that UNITS_DF contains expected data for a known unit."""
        row = UNITS_DF[UNITS_DF["symbol"] == "mV"]
        with subtests.test("mV unit exists"):
            assert not row.empty

        if not row.empty:
            with subtests.test("mV unit name"):
                assert row.iloc[0]["name"] == "milliVolt"
            with subtests.test("mV unit description"):
                assert row.iloc[0]["description"] == "Milli Unit of electric potential"


class TestCaseInsensitiveMatching:
    """Test class for case-insensitive unit matching logic.

    Tests the two-stage matching strategy:
    1. Exact symbol match (case-sensitive, preserves SI prefix distinction)
    2. Case-insensitive fallback (handles inconsistent capitalization)
    """

    @pytest.fixture
    def exact_match_test_cases(self):
        """Test cases requiring exact symbol matching (stage 1)."""
        return [
            # SI prefixes are case-sensitive
            {"input": "mV", "expected_name": "milliVolt", "expected_symbol": "mV"},
            {"input": "MV", "expected_name": "megaVolt", "expected_symbol": "MV"},
            {"input": "nT", "expected_name": "nanoTesla", "expected_symbol": "nT"},
            {"input": "km", "expected_name": "kilometer", "expected_symbol": "km"},
            {"input": "pV", "expected_name": "picoVolt", "expected_symbol": "pV"},
        ]

    @pytest.fixture
    def case_insensitive_fallback_test_cases(self):
        """Test cases requiring case-insensitive fallback (stage 2)."""
        return [
            # Base units with incorrect capitalization
            {"input": "M", "expected_name": "meter", "expected_symbol": "m"},
            {"input": "v", "expected_name": "Volt", "expected_symbol": "V"},
            {"input": "t", "expected_name": "Tesla", "expected_symbol": "T"},
            {"input": "ohm", "expected_name": "Ohm", "expected_symbol": "\u03a9"},
        ]

    @pytest.fixture
    def compound_unit_test_cases(self):
        """Test cases for compound units with mixed case."""
        return [
            # Original issue: 'V/M' should parse as Volt per meter
            {
                "input": "V/M",
                "expected_name": "Volt per meter",
                "expected_symbol": "V/m",
            },
            # Lowercase variant
            {
                "input": "v/m",
                "expected_name": "Volt per meter",
                "expected_symbol": "V/m",
            },
            # Mixed case with prefixes
            {
                "input": "mV/M",
                "expected_name": "milliVolt per meter",
                "expected_symbol": "mV/m",
            },
            # Complex compound unit
            {
                "input": "mV nT/[km ohm]",
                "expected_name": "milliVolt nanoTesla per kilometer Ohm",
                "expected_symbol": "mV nT/km \u03a9",
            },
        ]

    def test_exact_symbol_matching(self, subtests, exact_match_test_cases):
        """Test that exact symbol matches take priority (stage 1)."""
        for case in exact_match_test_cases:
            with subtests.test(input_unit=case["input"]):
                unit = get_unit_from_df(case["input"])
                assert (
                    unit.name == case["expected_name"]
                ), f"Expected '{case['expected_name']}' but got '{unit.name}'"
                assert unit.symbol == case["expected_symbol"]

    def test_case_insensitive_fallback(
        self, subtests, case_insensitive_fallback_test_cases
    ):
        """Test that case-insensitive matching works for base units (stage 2)."""
        for case in case_insensitive_fallback_test_cases:
            with subtests.test(input_unit=case["input"]):
                unit = get_unit_from_df(case["input"])
                assert (
                    unit.name == case["expected_name"]
                ), f"Expected '{case['expected_name']}' but got '{unit.name}'"
                assert unit.symbol == case["expected_symbol"]

    def test_compound_units_with_mixed_case(self, subtests, compound_unit_test_cases):
        """Test compound units that require both exact and case-insensitive matching."""
        for case in compound_unit_test_cases:
            with subtests.test(input_unit=case["input"]):
                unit = get_unit_object(case["input"])
                assert (
                    unit.name == case["expected_name"]
                ), f"Expected '{case['expected_name']}' but got '{unit.name}'"
                assert unit.symbol == case["expected_symbol"]

    def test_prefix_distinction_preserved(self, subtests):
        """Test that SI prefix case sensitivity is maintained."""
        test_pairs = [
            ("m", "k"),  # milli vs kilo prefix
            ("M", "k"),  # mega vs kilo prefix
            ("p", "n"),  # pico vs nano prefix
        ]

        for prefix1, prefix2 in test_pairs:
            with subtests.test(prefix_comparison=f"{prefix1} vs {prefix2}"):
                unit1 = get_unit_from_df(f"{prefix1}V")
                unit2 = get_unit_from_df(f"{prefix2}V")
                # Ensure they resolve to different units
                assert (
                    unit1.name != unit2.name
                ), f"Prefixes {prefix1} and {prefix2} should not match the same unit"
