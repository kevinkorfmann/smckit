from pathlib import Path

import smckit


def test_example_path_returns_readable_psmc_fixture(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = smckit.io.example_path("psmc/NA12878_chr22.psmcfa")

    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "NA12878_chr22.psmcfa"

    data = smckit.io.read_psmcfa(path)
    assert data.uns["sum_L"] > 0
    assert data.uns["sum_n"] > 0


def test_example_path_returns_packaged_smcpp_fixture(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = smckit.io.example_path("smcpp/example.smc.gz")

    assert path.exists()

    data = smckit.io.read_smcpp_input(path)
    assert data.uns["total_sites"] > 0
    assert data.uns["n_undist"] == 5


def test_example_prefix_materializes_asmc_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = smckit.io.example_prefix(
        "asmc/exampleFile.n300.array",
        (".hap.gz", ".samples", ".map.gz"),
    )

    assert Path(str(root) + ".samples").exists()
    assert Path(str(root) + ".map.gz").exists()
    assert Path(str(root) + ".hap.gz").exists()
