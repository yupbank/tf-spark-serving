import tensorflow as tf
import pyspark.sql.functions as F
import pyspark.sql.types as T

from copy_op import eval_graph
from utils import tf_shape_as_array, get_model_full_path


def schema_infer(tensor_mappings):
    """
    Generate spark schema from tensor mappings.

    We skip the first dimension intensionally, so a tensor would fit in a spark dataframe.
    """
    return T.StructType([
        T.StructField(name,
                      tf_shape_as_array(tensor.shape.as_list()[1:],
                                        TF_TO_SPARK_TYPE_MAPPINGS[tensor.dtype]))
        for name, tensor in tensor_mappings.items()
    ])


def load_model(model_base_path, model_version=None):
    model_full_path = get_model_full_path(model_base_path, model_version)
    return tf.contrib.predictor.saved_model_predictor.SavedModelPredictor(model_full_path)


def function_wrapper(model):
    output_schema = schema_infer(model._fetch_tensors)

    def model_fn(features, labels, mode, params, config):
        to_graph = list(features.values)[0].graph
        input_mappings = {model._feed_tensors[key]: value
                          for key, value in features.iterms()}
        keys = model._fetch_tensors.keys()
        outputs = [model._fetch_tensors[key] for key in keys]
        predictions = dict(zip(keys,
                               eval_graph(input_mappings,
                                          outputs,
                                          to_graph)))
        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                          predictions=predictions)

    def func(df):
        tf_input_func = tf.estimator.inputs.pandas_input_fn(x=df, shuffle=False)
        est = tf.estimator.Estimator(model_fn=model_fn, model_dir=None)
        return pd.Dataframe(est.predict(tf_input_func, config))

    return F.pandas_udf(func, output_schema, F.PandasUDFType.GROUPED_MAP)


def tf_batch_inference(df, model_base_path, model_version=None):
    model = load_model(model_base_path, model_version)
    inferecer = function_wrapper(model)
    return df.groupby().apply(inferecer)
