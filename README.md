# Purpose

This repository exists for my experiments on BlueBERT. As the model currently (in June 2022) achieving state of the art results in integrating the UMLS medical knowledge base into the BERT neural network, I will use it as benchmark for another model.

# Installation

Go to your shell and type these commands :
```
# Enter the folder where you want the code to be stored
cd folder/to/store/the/code

# Get the code
git clone https://github.com/Gael-Bernard/BlueBERT_experiments
cd BlueBERT_experiments
```

* If you only want to execute the code and don't plan to add new dependencies, you can install dependencies using the hard requirements (dependency versions I use) :
   ```
   pip install -r hard_requirements.txt
   ```
* If you may need to add new dependencies, you'd better install loose requirements to reduce the risk of incompatibility :
   ```
   pip install -r requirements.txt
   ```

**TODO : Fill the requirements files**