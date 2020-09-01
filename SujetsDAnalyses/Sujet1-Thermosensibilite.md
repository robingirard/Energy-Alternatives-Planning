# Evolution de la Thermosensibilité

## Problématique
La thermosensibilité est la sensibilité de la consommation électrique à la température.
Pour connaitre sa définition et son évolution passée vous pouvez lire [ce post](https://www.energy-alternatives.eu/2019/05/24/variabilite-de-la-consommation-electrique-et-thermo-sensibilite/). 
Son évolution dans le contexte de la transition énergétique est un enjeu que j'ai discuté [ici](https://www.energy-alternatives.eu/2020/03/22/une-contribution-a-la-reflexion-sur-la-strategie-nationale-bas-carbone-dans-le-batiment-partie-1-quels-modes-de-chauffage-a-lhorizon-2050/) sans vraiment intégrer une modélisation des impacts de cette thermosensibilité sur le système électrique. 

## Objectifs

Dans ce projet il s'agit de comprendre les implication pour le système électrique de l'évolution de la thermosensibilité. On peut pour cela faire des scénarios d'évolution de la thermosensibiltié, et de choix politiques (e.g. renouvelable ou nucléaire, contrainte sur le CO2 ou juste taxe sur le CO2) et regarder comment ils impactent le coût complet du système électrique. 

Pour une modélisation plus profonde du système il faut intégrer l'origine de la thermosensibilité. Pour aller plus loin dans ce projet, vous pouvez donc reprendre la [discussion sur l'évolution des modes de chauffage]((https://www.energy-alternatives.eu/2020/03/22/une-contribution-a-la-reflexion-sur-la-strategie-nationale-bas-carbone-dans-le-batiment-partie-1-quels-modes-de-chauffage-a-lhorizon-2050/)) mentionnée précédement, 
mais en intégrant une évaluation du coût du système électrique. 
Ceci vous permettra de proposer une optimisation de l'ensemble "changement de mode de chauffage + évolution du système électrique". Cette optimisation pourra être faite simplement en évaluant le coût de plusieurs plusieurs cas ou en utilisant des règles de bon sens. 
Vous pouvez aussi envisager de faire cette optimisation de manière formelle en l'intégrant dans la modélisation pyomo. Ici il peut-être intéressant de discuter les résultats selon les choix politiques qui sont fait sur le système (e.g. renouvelable vs nucléaire).
Dans vos discussions vous pourrez imaginer un système réglementaire/économique (taxes, primes, réglementation thermiques, marché de capacité, ...) qui permette de faire advenir le meilleur système. 

## Outils à disposition 

Toutes les fonctions qui permette de décomposer la consommation en consommation thermosensible et consommation non thermosensible. Elles se trouvent dasn "functions_decompose_thermosensibilite.py" et des exemples d'utilisation sont donnés dans la correction de la question 1 "Question_1_TP.py". 