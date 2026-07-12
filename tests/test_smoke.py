from __future__ import annotations

import myProject


def testVersionIsSet():
    assert isinstance(myProject.__version__, str)
    assert myProject.__version__
