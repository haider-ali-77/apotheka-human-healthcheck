import requests
import pandas as pd
import os
data_dir = '../data/'
sample_csv = 'sample_data.csv'
csv_data_path = os.path.join(data_dir,'vaSuicidePreventionInnovation.csv')
df = pd.read_csv(csv_data_path)
df = df.dropna()

dataset = []
feature_names = []
for column in df.columns:
    feature_names.append(column)
    vals = df[column].values.tolist()
    dataset.append(vals)

# input_dict = {'dataset':dataset, 'plot_requirements':{'image_width':720, 'image_height':1280}, 'feature_names':feature_names}
input_dict = {'dataset':dataset, 'plot_requirements':{'image_width':1280, 'image_height':720}}
r = requests.post('http://127.0.0.1:5001/dataset',json=input_dict)
