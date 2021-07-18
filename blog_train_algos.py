import os
os.system('pip install pandas')
os.system('pip install gluonts')
import pandas as pd
import pathlib
import gluonts
import numpy as np
import argparse
import json
import boto3
from mxnet import gpu, cpu
from mxnet.context import num_gpus
from gluonts.dataset.util import to_pandas
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions, backtest_metrics
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from smdebug.mxnet import Hook

s3 = boto3.client("s3")

def uploadDirectory(model_dir,prefix,bucket):
    for root,dirs,files in os.walk(model_dir):
        for file in files:
            print(os.path.join(root,file))
            print(prefix+file)
            s3.upload_file(os.path.join(root,file),bucket,prefix+file)




def train(bucket, seq, algo, freq, prediction_length, epochs, learning_rate, hybridize, num_batches_per_epoch):
    
    #create train dataset
    df = pd.read_csv(filepath_or_buffer=os.environ['SM_CHANNEL_TRAIN'] + "/train.csv", header=0, index_col=0)
    
    training_data = ListDataset([{"start": df.index[0], 
                                  "target": df.usage[:],
                                 "item_id": df.client[:]}], 
                                  freq=freq)
    
    
    #create test dataset
    df = pd.read_csv(filepath_or_buffer=os.environ['SM_CHANNEL_TEST'] + "/test.csv", header=0, index_col=0)
    
    test_data = ListDataset([{"start": df.index[0], 
                              "target": df.usage[:],
                              "item_id": 'client_12'}], 
                              freq=freq)
    
    hook = Hook.create_from_json_file()
    #determine estimators##################################
    if algo == "DeepAR":
        estimator = DeepAREstimator(freq=freq,  
                                       prediction_length=prediction_length,
                                       context_length=1,
                                       trainer=Trainer(ctx="cpu",
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            hybridize=hybridize,
                                            num_batches_per_epoch=num_batches_per_epoch
                                         ))
    
        #train the model
        predictor = estimator.train(training_data=training_data)
        print("DeepAR training is complete SUCCESS")
    elif algo == "SFeedFwd":
        estimator = SimpleFeedForwardEstimator(freq=freq,  
                                       prediction_length=prediction_length,
                                       trainer=Trainer(ctx="cpu",
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            hybridize=hybridize,
                                            num_batches_per_epoch=num_batches_per_epoch
                                         ))
    
        #train the model
        predictor = estimator.train(training_data=training_data)
        print("training is complete SUCCESS")
    elif algo == "lstnet":
        # Needed for LSTNet ONLY
        grouper = MultivariateGrouper(max_target_dim=6)
        training_data = grouper(training_data)
        test_data = grouper(test_data)
        context_length = prediction_length
        num_series = 1
        skip_size = 1
        ar_window = 1
        channels = 4
        
        estimator = LSTNetEstimator(freq=freq,  
                                       prediction_length=prediction_length,
                                       context_length=context_length,
                                       num_series=num_series,
                                       skip_size=skip_size,
                                       ar_window=ar_window,
                                       channels=channels, 
                                       trainer=Trainer(ctx="cpu",
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            hybridize=hybridize,
                                            num_batches_per_epoch=num_batches_per_epoch
                                         ))
    
        #train the model
        predictor = estimator.train(training_data=training_data)
        print("training is complete SUCCESS")
    elif algo == "seq2seq":
        estimator = MQCNNEstimator(freq=freq,  
                                       prediction_length=prediction_length,
                                       trainer=Trainer(ctx="cpu",
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            hybridize=hybridize,
                                            num_batches_per_epoch=num_batches_per_epoch
                                         ))
    
        #train the model
        predictor = estimator.train(training_data=training_data)
        print("training is complete SUCCESS")
    else:
        estimator = TransformerEstimator(freq=freq,  
                                       prediction_length=prediction_length,
                                       trainer=Trainer(ctx="cpu",
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            hybridize=hybridize,
                                            num_batches_per_epoch=num_batches_per_epoch
                                         ))
    
        #train the model
        predictor = estimator.train(training_data=training_data)
        print("training is complete SUCCESS")
    
    ###################################################
    
    #evaluate trained model on test data
    forecast_it, ts_it = make_evaluation_predictions(test_data, predictor,  num_samples=100)
    print("EVALUATION is complete SUCCESS")
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
    print("METRICS retrieved SUCCESS")
    #bucket = "bwp-sandbox"
    
    mainpref = "gluonts/blog-models/"
    prefix = mainpref + str(seq) + "/"
    agg_df = pd.DataFrame(agg_metrics, index=[0])
    file = "metrics"+str(seq)+".csv"
    os.system('mkdir metrics')
    cspath = os.path.join('metrics', file)
    agg_df.to_csv(cspath)
    s3.upload_file(cspath,bucket,mainpref+"metrics/"+file)
    
    
    hook.save_scalar("MAPE", agg_metrics["MAPE"], sm_metric=True)
    hook.save_scalar("RMSE", agg_metrics["RMSE"], sm_metric=True)
    hook.save_scalar("MASE", agg_metrics["MASE"], sm_metric=True)
    hook.save_scalar("MSE", agg_metrics["MSE"], sm_metric=True)
    
    print("MAPE:", agg_metrics["MAPE"])
    
    #save the model
    predictor.serialize(pathlib.Path(os.environ['SM_MODEL_DIR'])) 
    
    
    uploadDirectory(os.environ['SM_MODEL_DIR'], prefix, bucket)
    
    return predictor





def model_fn(model_dir):
    path = pathlib.Path(model_dir)   
    predictor = Predictor.deserialize(path)
    
    return predictor


def transform_fn(model, data, content_type, output_content_type):
    
    data = json.loads(data)
    df = pd.DataFrame(data)

    test_data = ListDataset([{"start": df.index[0], 
                              "target": df.usage[:]}], 
                               freq=freq)
    
    forecast_it, ts_it = make_evaluation_predictions(test_data, model,  num_samples=100)  
    #agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(test_data))
    #response_body = json.dumps(agg_metrics)
    response_body = json.dumps({'predictions':list(forecast_it)[0].samples.tolist()[0]})
    return response_body, output_content_type


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, default="")
    parser.add_argument('--seq', type=str, default="sample")
    parser.add_argument('--algo', type=str, default="DeepAR")
    parser.add_argument('--freq', type=str, default='D')
    parser.add_argument('--prediction_length', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--hybridize', type=bool, default=True)
    parser.add_argument('--num_batches_per_epoch', type=int, default=10)    
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args.bucket, args.seq, args.algo, args.freq, args.prediction_length, args.epochs, args.learning_rate, args.hybridize, args.num_batches_per_epoch)
  
