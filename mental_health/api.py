import flask
from flask import request, jsonify
from  stats import VASuicide as statsVASuicide
from recommender import VASuicide as modelVASuicide
import os
from waitress import serve

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Create some test data for our catalog in the form of a list of dictionaries.

input_dataset = {}
stats_output_response = {}
data_dir = 'data'
mental_health_indicator_path = os.path.join(data_dir,'vaSuicidePreventionInnovation.ods')


@app.route('/', methods=['GET'])
def home():
  return '''<h1>RAI API FOR VASUICIDE DATA STATS AND PREDICTIONS</h1>'''


@app.route('/dataset',methods=['POST'])
def api_put_dataset():
  global input_dataset
  input_dataset =  request.get_json()
  return 'Done'


@app.route('/stats', methods=['GET'])
def api_get_stats():
  apotheka = statsVASuicide(input_dataset , data_dir, correlation='compute')
  apotheka.quality_stats()
  apotheka.plots()
  stats_output_response = apotheka.get_output_response()
  return jsonify(stats_output_response)

@app.route('/train', methods=['GET'])
def api_train():
  model = modelVASuicide(mental_health_indicator_path,input_dataset)
  logs = model.train('models','gpu')
  return jsonify(logs)

@app.route('/AIRecommender', methods=['GET'])
def api_recommender():
  model = modelVASuicide(mental_health_indicator_path,input_dataset)
  predictions = model('models')
  return jsonify(predictions)

if __name__=='__main__':
    #app.run(host='0.0.0.0')
    serve(app,host='0.0.0.0',port=5000)

