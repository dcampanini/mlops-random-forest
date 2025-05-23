# MLOps pipeline using Vertex AI on GCP

The files in this repository allow you to build a basic pipeline on GCP. \
Copy the next instruction in your terminal to successfully build a MLOps pipeline using Vertex AI.


 - Clone repository
```
git clone  https://github.com/dcampanini/mlops-random-forest.git
```
 - Create a Python virtual environment:
```
python3 -m venv env_mlops
```
- Activate the virtual environment recently created
```
source env_mlops/bin/activate
```
- Change to the local repository directory
```
cd mlops-random-forest
```
- Install the required libraries using the file requirements.txt
```
pip install -r requirements.txt
```
- Run the code to compile the pipeline
```
python pipeline_randomforest.py
```
- Run the code to execute the pipeline
```
python execute_pipeline.py
```