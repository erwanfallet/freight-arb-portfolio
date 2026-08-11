"""Agricultural arbitrage portfolio.

Three tiers, which differ **only in where the disagreement comes from**:

    T1  documented, quotable disagreement    "You said that…"
    T2  inferred structural tension          "It seems to me…"
    T3  disagreement open today              "Hedgepoint says A, Czarnikow says B"

The method is identical in all three cases:
    1. an explicit accounting identity, rebuilt term by term;
    2. a binary regime — open/shut, above/below;
    3. a numeric tipping point: that's the deliverable;
    4. a question only an insider can settle.

Absolute rule on T2: never present an inferred tension as a quotation.
Every T2 engine carries this warning in its docstring.
"""

__version__ = "0.1.0"
