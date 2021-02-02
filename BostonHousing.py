import kfp
from kfp import dsl
from kfp import compiler
import kfp.components as comp
from kfp import gcp

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='gcr.io/kubeflow-demos/preprocess:latest',
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy',
        }
    )


def train_op(x_train, y_train):

    return dsl.ContainerOp(
        name='Train Model',
        image='gcr.io/kubeflow-demos/train-2:latest',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train
        ],
        file_outputs={
            'model': '/app/model.pkl'
        }
    )

def upload_op(model):
    
    return dsl.ContainerOp(
        name='upload to GCS',
        image='gcr.io/kubeflow-demos/upload:latest',
        arguments=[
            '--model', model
        ]
    )

def test_op(x_test, y_test, model):

    return dsl.ContainerOp(
        name='Test Model',
        image='gcr.io/kubeflow-demos/test:latest',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ],
        file_outputs={
            'mean_squared_error': '/app/output.txt',
            'mlpipeline-metrics': '/mlpipeline-metrics.json'
        }
    )

def deploy_model_op(model):

    return dsl.ContainerOp(
        name='Deploy Model',
        image='gcr.io/kubeflow-demos/deploy:latest',
        arguments=[
            '--model', model
        ]
    )


# In[200]:


@dsl.pipeline(
   name='Boston Housing Pipeline',
   description='Build Scikit model to predict house prices'
)
def boston_pipeline(model_version: str):
    _preprocess_op = preprocess_op()
    _preprocess_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train'])
    ).after(_preprocess_op)
    _train_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    
    _upload_op = upload_op(
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)
    _upload_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    _test_op = test_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)
    _test_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    deploy_op =deploy_model_op(
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_test_op)
    deploy_op.execution_options.caching_strategy.max_cache_staleness = "P0D"


# In[204]:


args = {
    "model_version": "11"
}

client = kfp.Client(host='c4e37c713669144-dot-us-central2.pipelines.googleusercontent.com')
client.create_run_from_pipeline_func(boston_pipeline, experiment_name='user-group-demo-experiment', arguments=args)


# In[ ]:




