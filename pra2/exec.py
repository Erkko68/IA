import subprocess
import os

# Get the absolute paths of the scripts
classification_script = os.path.abspath('./src/classification/classification.py')
clustering_script = os.path.abspath('./src/clustering/clustering.py')
regression_script = os.path.abspath('./src/regression/regression.py')


# Execute clustering script
print("Starting execution clustering.py")
subprocess.run(['python3', clustering_script], cwd=os.path.dirname(clustering_script), check=True)
print("Completed execution of clustering.py")

# Execute classification script
print("Moving to the classification step...")
subprocess.run(['python3', classification_script], cwd=os.path.dirname(classification_script), check=True)
print("Completed execution of classification.py")

# Execute regression script
print("Moving to the regression step...")
subprocess.run(['python3', regression_script], cwd=os.path.dirname(regression_script), check=True)
print("Completed execution of regression.py")

print("All scripts executed successfully.")
