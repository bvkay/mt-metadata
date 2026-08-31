# -*- coding: utf-8 -*-
"""
Regression tests for reading an EDI that declares a single frequency.

`EDI._assert_descending_frequency` compared ``frequency[0]`` with
``frequency[1]`` without checking how many frequencies were present, so a
well-formed ``NFREQ=1`` file raised ``IndexError: index 1 is out of bounds for
axis 0 with size 1`` before the transfer function was built. A single
frequency is trivially ordered, so there is nothing to reorder and nothing to
compare.

Single-period EDIs occur wherever a transfer function is recovered from a
published table that printed one period per site, which is common in older
geomagnetic depth sounding arrays.
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pytest

from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.io import edi

# =============================================================================
# A minimal well-formed tipper-only EDI with NFREQ=1
# =============================================================================
SINGLE_PERIOD_EDI = """>HEAD

  DATAID="SP1"
  ACQBY="test"
  LAT=-37.316700
  LONG=143.000000
  EMPTY=1.0E+32

>INFO   MAXLINES=1000

  A single-period transfer function.

>=DEFINEMEAS

  MAXCHAN=3
  MAXRUN=999
  MAXMEAS=99999
  UNITS=M
  REFTYPE=CART
  REFLAT=-37.316700
  REFLONG=143.000000

 >HMEAS ID= 1001.001 CHTYPE=HX X =  .000000  Y =  .000000  AZM=   0.0000
 >HMEAS ID= 1002.001 CHTYPE=HY X =  .000000  Y =  .000000  AZM=  90.0000
 >HMEAS ID= 1003.001 CHTYPE=HZ X =  .000000  Y =  .000000  AZM=  .000000

>=MTSECT
  SECTID=SP1
  NFREQ=    1
  HX = 1001.001
  HY = 1002.001
  HZ = 1003.001

 >!****FREQUENCIES****!
 >FREQ NFREQ=   1  ORDER=DEC //    1
    0.20833E-03
 >!****TIPPER PARAMETERS****!
 >TXR.EXP ROT=0.0 //    1
    0.55000E+00
 >TXI.EXP ROT=0.0 //    1
    0.14000E+00
 >TXVAR.EXP ROT=0.0 //    1
    0.90000E-01
 >TYR.EXP ROT=0.0 //    1
   -0.12000E+00
 >TYI.EXP ROT=0.0 //    1
    0.90000E-01
 >TYVAR.EXP ROT=0.0 //    1
    0.90000E-01

>END
"""


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def single_period_fn(tmp_path_factory):
    """Write the single-period EDI once for all tests in this module."""
    fn = tmp_path_factory.mktemp("single_period") / "sp1.edi"
    fn.write_text(SINGLE_PERIOD_EDI)
    return fn


@pytest.fixture(scope="module")
def single_period_edi_object(single_period_fn):
    return edi.EDI(fn=single_period_fn)


@pytest.fixture(scope="module")
def single_period_tf_object(single_period_fn):
    tf_obj = TF(fn=single_period_fn)
    tf_obj.read()
    return tf_obj


# =============================================================================
# Tests
# =============================================================================
def test_read_single_period_edi(single_period_edi_object):
    """A one-frequency EDI reads instead of raising IndexError."""
    assert single_period_edi_object.frequency.size == 1
    assert single_period_edi_object.frequency[0] == pytest.approx(0.20833e-03)


def test_single_period_tipper(single_period_edi_object):
    """The one estimate the file carries survives the read."""
    assert single_period_edi_object.t.shape == (1, 1, 2)
    assert single_period_edi_object.t[0, 0, 0] == pytest.approx(0.55 + 0.14j)
    assert single_period_edi_object.t[0, 0, 1] == pytest.approx(-0.12 + 0.09j)


def test_single_period_tf(single_period_tf_object):
    """The same through the TF layer."""
    assert single_period_tf_object.period.size == 1
    assert single_period_tf_object.has_tipper()
    assert not single_period_tf_object.has_impedance()


def test_assert_descending_frequency_is_a_no_op_for_one_frequency():
    """Called directly: one frequency is already ordered, so nothing changes."""
    edi_object = edi.EDI()
    edi_object.frequency = np.array([10.0])
    edi_object._assert_descending_frequency()
    assert edi_object.frequency == np.array([10.0])


def test_assert_descending_frequency_still_reorders_two_frequencies():
    """The behaviour the guard must not disturb: an ascending pair is flipped."""
    edi_object = edi.EDI()
    edi_object.frequency = np.array([1.0, 10.0])
    edi_object._assert_descending_frequency()
    assert np.array_equal(edi_object.frequency, np.array([10.0, 1.0]))
