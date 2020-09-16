# Etude_TP_CapaExpPlaning-Python

Ce projet contient le code et les données pour faire le TP de simulation du système électrique.


### Table des matières

* [1. Installations préalables](#installations)
* [2. Repertoire CSV](#CSV)
* [3. Repertoire CasDEtude](#CasDEtude)
* [4. Repertoire TP](#TP)
* [5. Repertoire functions](#functions)

## 1 - Installations préalables <a class="anchor" id="1.introduction"></a>

    conda env create -f conda.yml
    conda activate Etude_TP
    
Si vous utilisez PyCharm vous devez fixer l'environnement dans les préférences "python interpreter" il faut 
## 2 - Repertoire CSV <a class="anchor" id="CSV"></a>
contient des données .csv pour le projet (production éolienne, consommarion, disponibilité nucléaire, données technico-économiques), ces fichiers sont chargés dans les différents codes "run"

## 3- Repertoire CasDEtude <a class="anchor" id="CasDEtude"></a>
contient les fichiers Jupyter Notebook qui permettent de simuler la planification et l'opération du système électrique avec et sans stockage

## 4- Repertoire TP <a class="anchor" id="TP"></a>
contient les fichiers Jupyter Notebook présentant les enjeux des différentes parties du TP ainsi quelques questions auxquelles il faut répondre. On y trouve également les scripts Python regroupant les questions et réponses des  quatre parties du TP

## 5- Repertoire functions <a class="anchor" id="functions"></a>
contient les fonctions utilisées pour la planification et l'opération du système électrique ainsi que pour la manipulation de la thermosensibilité
