"""Real-data loading, outside the engines and outside the synthetic fixtures.

This module runs no thesis calculation: it reads the user's Bloomberg export and
returns clean pd.Series, indexed by date, with the unit explicitly declared.
"""
