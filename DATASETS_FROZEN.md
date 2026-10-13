# Frozen datasets

Written before any model was run and not changed afterwards. Choosing or
swapping datasets after seeing cascade results is the easiest way to make an
honest study unpublishable, and it is invisible without a pre-commitment. This
file plus `data/frozen/manifest.json` is that commitment, and every later stage
verifies its inputs against the manifest checksums.

## The four datasets

| Dataset | Domain | Labels | Frozen | Evaluated | Max tokens | Truncated |
|---|---|---|---|---|---|---|
| CFPB | Consumer finance complaints | 10 | 2000 | 1000 | 512 | 11.1% |
| Banking77 | Banking intents | 77 | 1925 | 1000 | 256 | 0.0% |
| LEDGAR | Contract provisions | 99 | 2000 | 500 | 512 | 1.9% |
| CLINC150 | Assistant intents | 149 | 1950 | 500 | 256 | 0.0% |

The label-space ladder runs 10 → 77 → 99 → 149. Input length runs the other way:
CLINC150 is one-line queries against a label list that fills almost the whole
prompt, while CFPB is long narratives against ten short product names. Those two
axes are deliberately not aligned, so a result that tracked label count alone
would be distinguishable from one that tracked prompt composition.

## Sources

**Banking77** comes from PolyAI's GitHub CSV, not the HuggingFace mirror. The
mirror ships a loading script that `datasets` 4.x refuses to execute, and that
same script blocked the hub's parquet auto-conversion, so `refs/convert/parquet`
does not exist for it either. The CSVs carry string labels, which removes any
integer-mapping risk.

**CLINC150** is `DeepPavlov/clinc_oos`, config `plus`, test split. The
out-of-scope class is dropped and the task stays closed-set over 150 intents.
That class is 1000 of 5500 rows, so the decision is not cosmetic. A label whose
name describes what an item is *not* gives a closed-set scorer nothing to
condition on, and keeping it would mix an abstention result into a routing
result without saying so.

**LEDGAR** is `coastalcph/lex_glue`, config `ledgar`, test split, via the
LexGLUE distribution.

**CFPB** is pulled from the public complaint API, narratives only, for a single
month (2024-01-01 to 2024-02-01). One month keeps the product taxonomy
internally consistent: the CFPB revised its product categories over time, and
mixing eras produces near-duplicate labels for the same concept, which is
unfixable after sampling and would read as model confusion in the results. One
product had too few narratives in that month to fill an equal quota and was
dropped.

Two API quirks are worth recording. The `format` parameter accepts only `csv`;
omitting it returns JSON, and passing `format=json` returns a 404. The CSV export
and the JSON search endpoint use different column names for the same fields,
title-case in one and snake_case in the other, so column names are resolved
case-insensitively rather than hard-coded.

## Sampling

Stratified by class with seed 0. Equal quota per class, except LEDGAR, where six
provision types hold fewer instances than an equal quota would need. There the
quota is a cap and the remainder is drawn from the natural distribution, which
keeps all 100 classes present. Label-space size is one of the axes the study
varies, so dropping the thin classes would have changed the thing being measured.

Frozen sample sizes are not all exactly 2000 because equal quotas do not divide
evenly into 2000 for 77, 99 or 149 classes. Balanced classes matter more to the
analysis than a round total, so the quotas were kept equal and the totals
allowed to vary. Per-dataset n is reported in every results table.

## Truncation

Set per dataset from the measured token-length distribution rather than by a
single global constant.

| Dataset | p50 | p90 | p95 | p99 | Limit | Cut |
|---|---|---|---|---|---|---|
| Banking77 | 15 | 29 | 35 | 48 | 256 | 0.0% |
| CLINC150 | 13 | 24 | 29 | 39 | 256 | 0.0% |
| LEDGAR | 95 | 273 | 366 | 603 | 512 | 1.9% |
| CFPB | 158 | 552 | 806 | 1807 | 512 | 11.1% |

CFPB at 512 cuts 11.1%, and raising it to 1024 would still cut 3.2% while
roughly doubling that dataset's inference cost relative to the others. Since the
cost model charges tokens, a longer limit on one dataset would muddy the
cross-dataset cost comparison for reasons unrelated to routing. Inspection of
long narratives showed the product is usually identifiable from the opening
sentences. Truncation applies identically to both tiers, so it cannot bias the
gap between them, which is the quantity the study measures.

## Preprocessing

Underscored label identifiers become natural language (`card_arrival` →
`card arrival`) for Banking77 and CLINC150. LEDGAR labels are already
title-cased prose and pass through unchanged.

CFPB narratives keep their `XXXX` redaction tokens exactly as published. There
are about 17.7 per narrative, so removing them was tested: collapsing runs to a
placeholder *raised* truncation to 11.5% because the placeholder tokenises
longer than the runs it merged, and dropping them entirely reached 9.6% but
mangled dates such as `XX/XX/2023` into `/ /2023`. A 1.5-point truncation
improvement was not worth a preprocessing step that needs defending, and the
redactions are part of what the published data is.

## Provenance of the frozen files

`data/frozen/manifest.json` records, per dataset: row count, label count,
truncation limit, evaluation size, sampling rule, any short classes, and a
sha256 prefix of the parquet file. `load_frozen()` recomputes that checksum on
every read and refuses to proceed if it has changed, because re-freezing would
silently invalidate every result already computed against the old sample.
