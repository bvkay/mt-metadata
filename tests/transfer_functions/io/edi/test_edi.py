# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:42:55 2021

@author: jpeacock
"""

import numpy as np

# =============================================================================
#
# =============================================================================
import pytest
from loguru import logger
from pydantic import HttpUrl

from mt_metadata.transfer_functions import TF
from mt_metadata.transfer_functions.io.edi import EDI
from mt_metadata.transfer_functions.io.edi.metadata import EMeasurement, HMeasurement
from mt_metadata.utils.validators import validate_doi


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def e_measurement_data():
    """Fixture to provide EMeasurement test data."""
    return {
        "id": 14.001,
        "azm": 0,
        "chtype": "EX",
        "x": -50.0,
        "y": 0.0,
        "x2": 50.0,
        "y2": 0.0,
    }


@pytest.fixture(scope="module")
def e_measurement_azm_data():
    """Fixture to provide EMeasurement azimuth test data."""
    return {
        "id": 14.001,
        "azm": 0,
        "chtype": "EX",
        "x": -50.0,
        "y": 0.0,
        "x2": 50.0,
        "y2": 10.0,
    }


@pytest.fixture(scope="module")
def h_measurement_data():
    """Fixture to provide HMeasurement test data."""
    return {
        "id": 12.001,
        "chtype": "HY",
        "x": 0.0,
        "y": 0.0,
        "azm": 90,
    }


@pytest.fixture(scope="module")
def edi_after_frequency_assert():
    """Fixture to create EDI object after asserting descending frequency."""
    edi = EDI()
    nf = 7
    edi.frequency = np.logspace(-3, 3, nf)
    z = np.arange(nf * 4).reshape((nf, 2, 2))
    t = np.arange(nf * 2).reshape((nf, 1, 2))
    edi.z = z.copy()
    edi.z_err = z.copy()
    edi.t = t.copy()
    edi.t_err = t.copy()

    # Store original arrays before transformation
    original_z = z.copy()
    original_t = t.copy()

    # Apply the frequency assertion
    edi._assert_descending_frequency()

    return {"edi": edi, "original_z": original_z, "original_t": original_t}


# =============================================================================
# EMeasurement Tests
# =============================================================================
class TestEMeasurement:
    """Test EMeasurement class functionality."""

    def test_attributes(self, e_measurement_data):
        """Test EMeasurement attributes."""
        ex = EMeasurement(**e_measurement_data)

        for k, v in e_measurement_data.items():
            actual = getattr(ex, k)
            if isinstance(v, (float, int)):
                assert (
                    abs(actual - v) < 1e-10
                ), f"Attribute {k} mismatch: expected {v}, got {actual}"
            else:
                assert (
                    actual == v
                ), f"Attribute {k} mismatch: expected {v}, got {actual}"


class TestEMeasurementAZM:
    """Test EMeasurement azimuth calculation."""

    def test_azimuth_calculation(self, e_measurement_azm_data):
        """Test EMeasurement azimuth calculation."""
        ex = EMeasurement(**e_measurement_azm_data)

        for k, v in e_measurement_azm_data.items():
            actual = getattr(ex, k)
            if k != "azm":
                assert (
                    actual == v
                ), f"Attribute {k} mismatch: expected {v}, got {actual}"
            else:
                expected = 5.7105931374996
                assert (
                    abs(actual - expected) < 1e-10
                ), f"Azimuth mismatch: expected {expected}, got {actual}"


# =============================================================================
# HMeasurement Tests
# =============================================================================
class TestHMeasurement:
    """Test HMeasurement class functionality."""

    def test_attributes(self, h_measurement_data):
        """Test HMeasurement attributes."""
        hy = HMeasurement(**h_measurement_data)

        for k, v in h_measurement_data.items():
            actual = getattr(hy, k)
            assert actual == v, f"Attribute {k} mismatch: expected {v}, got {actual}"


# =============================================================================
# Descending Frequency Tests
# =============================================================================
class TestAssertDescendingFrequency:
    """Test assert descending frequency functionality."""

    @pytest.mark.parametrize("array_type", ["z", "z_err", "t", "t_err"])
    def test_array_reversal(self, edi_after_frequency_assert, array_type):
        """Test that arrays are properly reversed when asserting descending frequency."""
        data = edi_after_frequency_assert
        edi = data["edi"]
        actual_array = getattr(edi, array_type)

        if array_type.startswith("z"):
            original_array = data["original_z"]
        else:
            original_array = data["original_t"]

        expected_array = original_array[::-1]

        assert np.allclose(
            expected_array, actual_array
        ), f"{array_type} array not properly reversed"


class TestSurveyMetadataDoiHandling:
    """Test mapping survey DOI keys to DOI or URL fields."""

    @pytest.mark.parametrize(
        "value, expected_field",
        [
            ("https://doi.org/10.1234/test", "citation_dataset.doi"),
            ("10.1234/test", "citation_dataset.doi"),
            ("doi:10.1234/test", "citation_dataset.doi"),
            ("https://example.com/dataset", "citation_dataset.url"),
            (HttpUrl("https://doi.org/10.1234/test"), "citation_dataset.doi"),
            (HttpUrl("https://example.com/dataset"), "citation_dataset.url"),
        ],
    )
    def test_survey_doi_values_map_to_expected_field(self, value, expected_field):
        edi = EDI()
        edi.Info.info_dict["survey.citation_dataset.doi"] = value

        survey_metadata = edi.survey_metadata
        actual_value = survey_metadata.get_attr_from_name(expected_field)

        if expected_field.endswith("doi"):
            assert actual_value.unicode_string() == validate_doi(value).unicode_string()
        elif isinstance(value, HttpUrl):
            assert actual_value.unicode_string() == value.unicode_string()
        else:
            assert actual_value == value

        other_field = (
            "citation_dataset.url"
            if expected_field.endswith("doi")
            else "citation_dataset.doi"
        )
        assert survey_metadata.get_attr_from_name(other_field) in [None, "None"]


def _make_tf(with_tipper=False):
    """Small TF with a declination and unit impedance."""
    tf = TF()
    tf.station = "dec01"
    tf.station_metadata.location.latitude = 31.87
    tf.station_metadata.location.longitude = -7.93
    tf.station_metadata.location.declination.value = -1.07
    tf.station_metadata.location.declination.model = "IGRF-13"
    tf.station_metadata.location.declination.epoch = "2023.5"
    tf.period = np.logspace(-2, 2, 6)
    tf.impedance = np.ones((6, 2, 2), dtype=complex)
    tf.impedance_error = np.full((6, 2, 2), 0.1)
    if with_tipper:
        tf.tipper = np.full((6, 1, 2), 0.1 + 0.0j)
        tf.tipper_error = np.full((6, 1, 2), 0.01)
    return tf


class TestDeclinationRoundTrip:
    """Declination value, model and epoch survive an EDI write and read."""

    def test_tf_round_trip(self, tmp_path):
        fn = tmp_path / "dec01.edi"
        _make_tf().write(fn=fn, file_type="edi")

        lines = [ln.strip() for ln in fn.read_text().splitlines()]
        assert "DECLINATION=-1.07" in lines

        tf = TF(fn=fn)
        tf.read()
        dec = tf.station_metadata.location.declination
        assert dec.value == -1.07
        assert dec.model == "IGRF-13"
        assert dec.epoch == "2023.5"

    def test_tf_comment_round_trip(self, tmp_path):
        fn = tmp_path / "dec01.edi"
        tf = _make_tf()
        tf.station_metadata.location.declination.comments.value = "IGRF-13 at the site"
        tf.write(fn=fn, file_type="edi")

        tf_in = TF(fn=fn)
        tf_in.read()
        dec = tf_in.station_metadata.location.declination
        assert dec.value == -1.07
        assert dec.comments.value == "IGRF-13 at the site"


class TestRotationAngleBlocks:
    """A TROT block that differs from ZROT is reported and ZROT is kept."""

    @staticmethod
    def _write_rot(fn, zrot, trot):
        _make_tf(with_tipper=True).write(fn=fn, file_type="edi")
        out = []
        block = None
        for line in fn.read_text().splitlines():
            if line.startswith(">"):
                block = line[1:].split()[0].lower() if line[1:].split() else None
            elif block == "zrot":
                line = " ".join(f"{zrot:.6E}" for _ in line.split())
            elif block == "trot":
                line = " ".join(f"{trot:.6E}" for _ in line.split())
            out.append(line)
        fn.write_text("\n".join(out) + "\n")

    @staticmethod
    def _read_with_warnings(fn):
        messages = []
        sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            edi = EDI(fn=fn)
        finally:
            logger.remove(sink)
        return edi, [m for m in messages if "TROT" in m]

    def test_differing_trot_warns(self, tmp_path):
        fn = tmp_path / "rot_diff.edi"
        self._write_rot(fn, 5.0, 7.0)
        edi, warnings = self._read_with_warnings(fn)

        assert np.allclose(edi.rotation_angle, 5.0)
        assert len(warnings) == 1
        assert "TROT [7.0]" in warnings[0]
        assert "ZROT [5.0]" in warnings[0]

    def test_equal_trot_silent(self, tmp_path):
        fn = tmp_path / "rot_same.edi"
        self._write_rot(fn, 5.0, 5.0)
        edi, warnings = self._read_with_warnings(fn)

        assert np.allclose(edi.rotation_angle, 5.0)
        assert warnings == []


class TestAzimuthWriteTwice:
    """Writing the same TF twice gives the same HMEAS/EMEAS AZM."""

    @staticmethod
    def _azm(fn):
        azm = {}
        for line in fn.read_text().splitlines():
            if line.startswith((">HMEAS", ">EMEAS")):
                items = dict(
                    item.split("=", 1) for item in line.split()[1:] if "=" in item
                )
                azm[items["CHTYPE"].lower()] = float(items["AZM"])
        return azm

    def test_write_twice(self, tmp_path):
        tf = _make_tf()
        run = tf.station_metadata.runs[0]
        for comp in ["ey", "hy"]:
            run.get_channel(comp).measurement_azimuth = 90.0

        azm = []
        for name in ["first.edi", "second.edi"]:
            tf.write(fn=tmp_path / name, file_type="edi")
            azm.append(self._azm(tmp_path / name))

        assert azm[0]["ey"] == 90.0
        assert azm[0]["hy"] == 90.0
        assert azm[1] == azm[0]


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__])
