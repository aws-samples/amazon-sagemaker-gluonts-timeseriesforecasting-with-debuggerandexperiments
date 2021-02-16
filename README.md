## How to train, debug and run time series forecasting at scale with the GluonTS toolkit on Amazon SageMaker

Time series forecasting is an approach to predict future data values by analyzing the patterns and trends in past observations over time. Organizations across industries require time series forecasting for a variety of use cases including seasonal sales prediction, demand forecasting, stock price forecasting, weather forecasting, financial planning, and inventory planning. 

There are various cutting edge algorithms available for time series forecasting such as [DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html), [SeqtoSeq family](https://ts.gluon.ai/api/gluonts/gluonts.model.seq2seq.html) and [lstnet](https://ts.gluon.ai/api/gluonts/gluonts.model.lstnet.html) (Long- and Short-term Time-series network) etc. The Machine Learning (ML) process for time series forecasting is often time consuming, resource intensive and requires comparative analysis across multiple parameter combinations and datasets to reach the required precision and accuracy with your models. To determine the best model, developers and data scientists need to first select algorithms and hyper-parameters, build, configure, train and tune models, evaluate these models, compare metrics captured at training and evaluation time, visualize results, and repeat this process several times before choosing a model that works. Moreover, the infrastructure management associated with the scaling required at training time for such an iterative process may lead to undifferentiated heavy lifting for the developers and data scientists involved.

In this [notebook](https://github.com/aws-samples/amazon-sagemaker-gluonts-timeseriesforecasting-with-debuggerandexperiments/blob/main/Amazon%20Sagemaker%20GluonTS%20time%20series%20forecasting.ipynb), we show you how to address these challenges by providing an approach with detailed steps to setup and run time series forecasting models at scale using GluonTS on Amazon SageMaker. The Gluon Time Series (GluonTS) is a Python toolkit for probabilistic time series modeling, built around Apache MXNet. GluonTS provides utilities for loading and iterating over time series datasets, state of the art models ready to be trained, and building blocks to define your own models and quickly experiment with different solutions. We will first show you how to setup GluonTS on SageMaker using the MXNet estimator, then train multiple models using SageMaker Experiments, use SageMaker Debugger to mitigate suboptimal training, evaluate model performance, and finally generate time series forecasts. We will walk you through the following steps:
1.	Prepare the time series dataset
2.	Create the algorithm and hyper-parameters combinatorial matrix
3.	Setup the GluonTS training script
4.	Setup an Amazon SageMaker Experiment and Trials
5.	Create the MXNet estimator
6.	Setup Experiment with Debugger enabled to auto-terminate suboptimal jobs
7.	Train and validate models
8.	Evaluate metrics and select a winning candidate
9.	Run time series forecasts

### Prerequisites

* Setup your Amazon SageMaker Notebook Instance *

To set up your notebook, complete the following steps:

1. Onboard to Amazon SageMaker Studio with the quick start procedure (https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html).
2. When you create an AWS Identity and Access Management (http://aws.amazon.com/iam) (IAM) role to the notebook instance, be sure to specify access to Amazon Simple Storage Service (http://aws.amazon.com/s3) (Amazon S3). You can choose Any *S3 Bucket* or specify the S3 bucket you want to enable access to. You can use the AWS-managed policies AmazonSageMakerFullAccess (https://us-west-2.console.aws.amazon.com/iam/home#/policies/arn:aws:iam::aws:policy/AmazonSageMakerFullAccess$jsonEditor) to grant general access to Amazon SageMaker services. 

3. When user is created and is active, choose *Open Studio*.
4. On the Studio landing page, from the File drop-down menu, choose New.
5. Choose Terminal.
6. In the terminal, enter the following code:

git clone https://github.com/aws-samples/amazon-sagemaker-gluonts-timeseriesforecasting-with-debuggerandexperiments

7. Open the notebook Amazon Sagemaker GluonTS time series forecasting.ipynb



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

