
This folder contains examples for basic functionalities : 

 - extraction, manipulation tools, and analysis of input data 
    - Availability of different production means (nuke, wind power, photovoltaic, hydro)
    - Consumption  (thermal sensitivity tools, electric vehicle, sectorial decomposition)
    - Economic data (operation cost, investment cost, technical constraints, 
 - optimisation 
    - Operation, 
    - a separate file for storage operation, 
    - and planing)

## Fichiers Operation_Singlezone.ipynb Operation_Multizone.ipypnb
Ces fichier permettent de simuler l'opération du système électrique et reposent sur l'utilisation de pyomo ainsi que les fonctions contenues dans functions_Operation.py
Ces fichiers permettent plus précisément grâce à Pyomo de construire un modèle à optimiser en rentrant :
- les sets : les moyens de production d'électricité et le temps
- les paramètres : la consommation électrique de la zone étudiée et le facteur de disponibilité de chaque moyen de production
- les variables : l'énergie produite par chaque moyen de production à l'instant t et le coût de production pour chaque moyen de production (qui correspondent aux valeurs à optimiser)
On utilise ensuite le solver Mosek pour résoudre ce problème d'optimisation.

## Fichiers Planing_Singlezone.ipynb Planing_Multizone.ipynb
permet de simuler la planification du système électrique. Repose sur l'utilisation de pyomo et des fonctions contenues dans functions_Planing.py

## Fichier Storage-Optim.ipynb
Permet de simuler l'utilisation de un ou deux stockage pour arbitrer des prix. Nécessite d'installer le package dynprogstorage, ce qui se fait normalement simplement en allant dans le répertoire dynprogstorage avec un terminal et en executant.

## Fichiers Operation_WithStorage_Singlezone.ipynb Planing_WithStorage_Singlezone.ipynb
Permet de simuler l'opération et la planification d'un système avec du stockage
il faut d'abord avoir compris Storage-Optim.ipynb
