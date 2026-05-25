# blip2-fine-tuning
Scripts created for bachelor's thesis at the Technical University of Košice

IMPORTANT
In the blip2_daquar_ft file csv_to_jsonl.py have to be placed in the dataset folder after the user downloads it from Kaggle (instructions are provided in the thesis pdf). So the final project structure should look like:
project folder - (blip2_daquar_ft in this case):
|-- dataset/
      |-- images/
      |-- csv and other dataset files downloaded from Kaggle
      |-- csv_to_jsonl.py
|-- scripts/
      |-- fine-tuning.py, evaluation.py, inference.py
|-- requirements.txt

Author's note:
comments in the code may contain minor mistakes (false statements or descriptions that conflict with actual code) due to a lot of edits applied while developing the code
