"""Chargement de données reelles, hors moteurs et hors fixtures synthetiques.

Ce module ne fait aucun calcul de these : il lit l'export Bloomberg de l'utilisateur et
rend des pd.Series propres, indexees par date, avec l'unite declaree explicitement.
"""
