"""Shared technical base for the agri pages — and reusable by the metals pages.

Written once, called everywhere. Contains the unit conversions, the resampling rules
that decide whether a test is valid, the statistics toolkit, and the tipping-point
solver that is every page's deliverable.

Nothing here knows about any particular commodity: `chains/` builds the models,
`core/` does the arithmetic and the statistics.
"""
