# TopicModeling-MLOps

## Configuration MLflow : Guide d'installation étape par étape

### 1. Installation de MLflow
Installez `mlflow` sur votre environnement Python:
\```
pip install mlflow
\```

### 2. Compatibilité et dépendances
Pour une installation complète incluant les dépendances courantes de MLflow, utilisez :
\```
pip install mlflow[extras]
\```

### 3. Configuration de la variable d'environnement PATH
Pour l'utilisation de l'exécutable `mlflow`, vous pourriez avoir besoin de mettre à jour votre `PATH`. Pour ce faire, ajoutez le chemin du dossier contenant l'exécutable `mlflow` à votre `~/.bashrc`:
\```
echo "export PATH=/home/ubuntu/.local/bin:$PATH" >> ~/.bashrc
\```
Si vous galérez parce que vous avez Windows et pas Linux, suivez les tutos YouTube, on peut trouver des exemples.

## Intégration et utilisation de MLflow : Guide étape par étape

### 1. Création d'une expérience MLflow
- **Préparation**:
  1. Créez un dossier nommé `MLflow`.
  2. Dans ce dossier, créez un fichier Python nommé `exemple.py` et insérez le code ci-dessous:

\```python
import mlflow

path = "/home/ubuntu/MLflow/mlruns"
mlflow.set_tracking_uri("file://" + path)
\```
Ensuite, assurez-vous de faire un start run avant l'entrainement de chaque modèle:
\```python
with mlflow.start_run(experiment_id=experiment_id):
\```

### 2. Traçage des données
- **Intégration de MLflow**:
  1. Intégrez MLflow à votre script pour suivre les données, entraînements, les hyperparamètres, les résultats et les artefacts.

- **Métriques et paramètres**:
  Utilisez `log_param` ou `log_metric` pour enregistrer individuellement des paramètres et des métriques. Pour un ensemble de valeurs, utilisez `log_params` et `log_metrics`. Notez que dans la documentation de MLflow, nous pouvons aussi faire un log de plein d'autres types d'objets.

Pour sauvegarder un modèle, utilisez `mlflow.log_model`. Dans la dernière version, cela ne marchait pas avec la librairie Gensim, alors on ne pourra peut-être pas le faire. À voir si cela change avec les nouvelles versions de MLflow.

\```python
fig.savefig("regression_plot.png")
plt.close(fig)

# artifacts (output files)
mlflow.log_artifact("hehe.png")
data.to_csv('italian_corpora_1997.txt', encoding='utf-8', index=False)
mlflow.log_artifact('italian_corpora_1997.txt')
\```

- **Structure des fichiers**:
  À l'exécution, un dossier `mlruns/` sera créé contenant les expériences MLflow. Chaque expérience aura son propre ID, et à l'intérieur, vous trouverez des dossiers pour chaque entraînement avec leurs métriques, paramètres, artefacts et tags, etc.

### 3. Utilisation de l'UI MLflow
- Pour visualiser vos entraînements dans l'interface graphique de MLflow:
  1. Ouvrez un terminal et naviguez vers le dossier contenant `exemple.py`.
  2. Exécutez la commande : `mlflow ui --host 0.0.0.0`, sinon allez directement sur vos navigateurs et allez vers l'UI de votre port exposé avec MLflow.
  3. Pour changer le port, ajoutez `--port VOTRE_PORT`.

**Note**: Assurez-vous de bien lancer `mlflow ui` dans un répertoire contenant le dossier `mlruns` depuis votre terminal...
