import os

# Install required packages
os.system("pip install pykeen networkx pandas numpy scikit-learn")
os.system("pip install spacy")
os.system("python -m spacy download en_core_web_sm")
os.system("pip install sentence_transformers")

# Install PyTorch Geometric and its dependencies
os.system("pip install torch-geometric")
os.system("pip install torch torchvision torchaudio")
os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
os.system("pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html")
os.system("pip install torch-geometric")
os.system("pip install SPARQLWrapper networkx torch-geometric")

print("All dependencies have been installed.")
