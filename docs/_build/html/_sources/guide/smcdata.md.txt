# The SmcData Container

Every smckit analysis revolves around the {class}`smckit.SmcData` container.
It is a single in-memory object that carries input sequences, model
parameters, and inference results through the entire analysis pipeline. This
design is borrowed directly from
[scanpy](https://scanpy.readthedocs.io)'s `AnnData` object: instead of
threading dozens of arrays through function arguments, you pass a single
`SmcData` object to each function, which mutates it in place and returns it.

## Anatomy of an SmcData

`SmcData` is a simple dataclass with five fields:

| Field | Type | Purpose |
|---|---|---|
| `sequences` | `np.ndarray \| None` | Input sequence data. Format depends on the method (binary heterozygosity for PSMC, haplotype matrix for MSMC, etc.). |
| `window_size` | `int` | Window size used for binning heterozygosity calls (default 100 bp). |
| `params` | `dict` | Model and analysis parameters: mutation rate, generation time, time discretization, etc. |
| `results` | `dict[str, dict]` | Inference results, keyed by method name (`"psmc"`, `"asmc"`, ...). |
| `uns` | `dict` | Unstructured annotations. Free-form storage for metadata, intermediate computations, or method-specific data that doesn't fit elsewhere. |

The `n_sites` property returns the number of sites/windows in `sequences`,
or `None` if no sequence data is loaded.

## The pipeline pattern

A typical smckit analysis follows three phases:

```python
import smckit

# 1. READ — io functions create or populate an SmcData
data = smckit.io.read_psmcfa("sample.psmcfa")

# 2. INFER — tl functions mutate the container, storing results
data = smckit.tl.psmc(data, pattern="4+5*3+4", n_iterations=25)

# 3. PLOT/ANALYZE — pl functions read from data.results
smckit.pl.demographic_history(data)
```

After this pipeline, the `data` object holds:

- `data.sequences` — the raw heterozygosity codes
- `data.params` — `mu`, `generation_time`, `pattern`, `max_t`, ...
- `data.results["psmc"]` — `theta`, `rho`, `lambda`, `ne`, `time_years`,
  `rounds`, ...
- `data.uns` — per-record metadata, EM diagnostics, anything internal to
  the method

You can then run a *second* method on the same container:

```python
data = smckit.tl.esmc2(data, n_iterations=20, estimate_beta=True)
# Now data.results has both "psmc" and "esmc2" keys
```

This is what enables side-by-side method comparison in a single
[`demographic_history`](plotting.md) plot.

## Which fields each method uses

Different SMC methods read and write different fields of `SmcData`. Here is a
high-level map (see the API reference for the exact dictionary keys each
method writes):

| Method | Reads from | Writes to `data.results[...]` |
|---|---|---|
| `tl.psmc` | `sequences`, `uns["records"]` | `"psmc"` |
| `tl.esmc2` | `sequences`, `uns["records"]` | `"esmc2"` |
| `tl.asmc` | `uns["haplotypes"]`, `uns["decoding_quantities"]` | `"asmc"` |
| `tl.msmc2` | `uns["multihetsep"]` | `"msmc2"` |
| `tl.msmc_im` | `uns["msmc_combined"]` (or input file path) | `"msmc_im"` |
| `tl.smcpp` | `uns["smcpp_input"]` | `"smcpp"` |
| `pl.demographic_history` | `data.results[method]`, `data.params` | (returns matplotlib `Axes`) |

## Comparison with AnnData

If you are coming from the scanpy ecosystem, the mapping is:

| AnnData | SmcData |
|---|---|
| `adata.X` | `data.sequences` |
| `adata.obs` / `adata.var` | (none — SMC data is not tabular) |
| `adata.uns` | `data.uns` |
| `adata.obsm` / `adata.varm` | (folded into `uns` and `results`) |
| (computed embeddings) | `data.results[method]` |

The biggest difference is that SMC analyses are inherently *single-sample
methods that produce per-method result dictionaries*, rather than tabular
single-cell objects. So instead of storing PCA results in `obsm["X_pca"]`,
smckit stores PSMC results in `results["psmc"]`.
