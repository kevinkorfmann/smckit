from __future__ import annotations

import warnings

from smckit._method_status import method_status, method_statuses
from smckit.tl._implementation import NativeTrustWarning, warn_if_native_not_trusted


def test_method_status_manifest_covers_public_methods() -> None:
    methods = {entry["method"] for entry in method_statuses()}
    assert {"psmc", "asmc", "msmc2", "msmc_im", "esmc2", "smcpp", "dical2", "ssm"} <= methods


def test_untrusted_native_method_emits_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_native_not_trusted("dical2", "native")
    assert any(isinstance(item.message, NativeTrustWarning) for item in caught)


def test_trusted_native_method_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_native_not_trusted("psmc", "native")
    assert not caught


def test_promoted_msmc_im_native_method_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_native_not_trusted("msmc_im", "native")
    assert not caught


def test_method_status_entries_include_docs_trust_flag() -> None:
    status = method_status("dical2")
    assert status["native_trusted_for_docs"] is False
    assert status["install_extra"] == "dical2"

    msmc_im_status = method_status("msmc_im")
    assert msmc_im_status["native_trusted_for_docs"] is True
    assert msmc_im_status["install_extra"] == "msmc_im"
