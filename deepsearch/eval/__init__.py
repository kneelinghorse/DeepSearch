"""Local-eval tooling (S69+).

Consumes cached mission outputs and produces evaluation signals for the
daily-eval cadence (S70). Strictly local — never wired into the production
runtime per ``cmos/reports/sprint_69_approach_and_plan.md`` PD10.

Imports are deferred to submodules (``deepsearch.eval.factscore``) so
``python -m deepsearch.eval.factscore`` and ``python -m deepsearch.eval``
both work without double-loading warnings.
"""
