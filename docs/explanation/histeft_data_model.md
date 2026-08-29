# HistEFT data and coefficient model

HistEFT stores a histogram whose bin content is a polynomial in Wilson
coefficients rather than one fixed number. The same object can therefore be
evaluated at the Standard Model (SM) point or at another coefficient point
without rerunning the event processor.

## What the object represents

For each physical histogram bin, HistEFT accumulates the coefficients of a
quadratic polynomial. The coefficient order is part of the data model: a
consumer must associate each stored term with the same pair of Wilson
coefficients used by the producer. Evaluating the histogram substitutes a set
of coefficient values into that polynomial and returns an ordinary numerical
histogram view.

The processor supplies event weights, category labels, dense observable
values, and the event-level EFT coefficient array. HistEFT owns accumulation
of those coefficients into histogram bins and the algebra needed for later
evaluation, slicing, grouping, addition, and serialization. It does not own
sample selection, event selection, artifact provenance, sumw2 policy, or the
downstream fit topology.

Nominal scalar content, EFT polynomial content, and sumw2 companions have
related but distinct storage and validation contracts. In particular, the
maintained sumw2 companion is a scalar second moment evaluated at the SM point;
it is not another EFT polynomial. Transformations and late bin aggregation
must preserve coefficient order and apply the same physical mapping to every
related nominal or companion object.

## Authority boundaries

The documentation separates three kinds of contract:

- The [HistEFT reference](../reference/histeft.md) owns the current class API,
  coefficient algebra, serialization behavior, current consumer requirements,
  and a clearly marked future replacement-parity design.
- The [histogram-artifact reference](../reference/histogram_artifacts.md) owns
  the top-level PKL, split nominal containers, transformation provenance, and
  compatibility schemas.
- The [sumw2 reference](../reference/sumw2.md) owns companion selection,
  provenance, coverage, and failure conditions.

This division matters because a class method, an artifact schema, and a
production policy evolve at different boundaries. A change to HistEFT
coefficient storage does not by itself redefine which sumw2 companions a
production must contain. Likewise, an artifact-sidecar version does not change
the polynomial evaluated by `HistEFT.eval`.

For a guided inspection of these objects, see the
[histogram tutorial](../tutorials/histogram_artifacts.md). For the repository-
wide PKL/sidecar boundary, see
[artifacts and provenance](artifacts_and_provenance.md).

## Why replacement parity is a separate design problem

A possible future histogram backend must reproduce more than method names.
Current processors and consumers depend on coefficient ordering, categorical
axis labels, flow behavior, pickle reconstruction, merging, grouping, and
evaluation at both the SM and nonzero coefficient points. Old PKLs may also
refer to the present module and class names during unpickling.

The parity fixtures and test matrix in the reference document are therefore a
design checklist for a future replacement, not a second runtime API or a claim
that a replacement has been chosen. They distinguish requirements needed by
current processors and consumers from optional compatibility work. Proposed
fixtures remain non-production examples until a separately authorized backend
change adopts and implements them.

Open questions about old-PKL conversion, legacy consumers, or a future
nonzero-Wilson-coefficient uncertainty model are similarly not current
behavior. The maintained contracts remain the current HistEFT implementation,
the checked-in artifact schemas, and the source-grounded consumer boundaries
linked above.
