# Tutorial

Des remarques et rappelles des commandes, instructions et autres choses qu'on a appris pendant la preparation pour le PSC.
N'oubliez pas, vous trouverez tous ça très rapidement sur Google s'il y a quelques questions. En français et en anglais aussi.

## Ligne de commande
Liste des commandes basiques pour la ligne de commande!
Les commandes sont donnés dans l'ordre shell/CMD. 
Sur windows il y a aussi l'option plus moderne d'utiliser PowerShell. Elle suit en majorité les commandes de la shell.

- `ls / dir` : lister tous les fichiers et dossiers du fichier present;
- `cd FOLDER_NAME` : ouvrir/rentrer dans un dossier;
- `cd ..` : revenir au dossier d’arrière;
- `mkdir FOLDER_NAME` : créer un nouveau dossier dans le dossier present;
- `touch FILE_NAME / echo > FILE_NAME` : créer un nouveau fichier;

## Python sur la ligne de commande
La bonne commande est soit `python` soit `python3`. Si vous en doutez pour votre machine, regarder la version utilisé (il faut que ça soit au moins 3.8)

- `python3 file.py` : execute le fichier sur le terminal;
- `python3 --version` : affiche la version de python utilisé;
- `python3 -m venv ENV_NAME` : crée un ambient virtual python dans le dossier;
- `source venv/bin/activate` : active l'ambient virtual; (vous allez voir le nom de l'ambient affiché toujours)
- `pip install PACKAGE_NAME` : installe un package soit sur l'ordi, soit sur l'ambient virtual (s'il est activé);

## Git sur la ligne de commande
N'OUBLIEZ JAMAIS DE FAIRE UN PULL AVANT DE COMMENCER A TRAVAILLER SUR QUELQUE CHOSE!!!!!
Et autres commandes usuelles de Git.

- `git clone REPO_URL`: cloner un repo dans le dossier present;
- `git pull`: télécharger les changements faites sur le repo en ligne;
- `git add FILE_NAME` ou `git add .` : ajoute le ficher (ou tous si vous mettez le `.`) localement;
- `git commit -m "MESSAGE"` : fait les modifications ajoutés dans le code (utilisez apres `git add`);
- `git push` : mettre vos alterations en ligne;