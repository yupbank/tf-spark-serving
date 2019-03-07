import tensorflow as tf

import pyspark.sql.types as T
from tensorflow.contrib.predictor.saved_model_predictor import SavedModelPredictor

TF_TO_SPARK_TYPE_MAPPINGS = {
    tf.float32: T.FloatType,
    tf.float64: T.DoubleType,
    tf.int32: T.IntegerType,
    tf.int64: T.LongType,
    tf.string: T.StringType
}


def tf_shape_as_array(shape, elem_type):
    if not shape:
        return elem_type()
    else:
        return T.ArrayType(_tf_shape_as_array(shape[1:], elem_type))


def get_model_full_path(model_base_path, version=None):
    versions = {}
    model_base_path = model_base_path.rstrip('/')
    for version in tf.gfile.ListDirectory(model_base_path):
        version = version.rstrip('/')
        if version.isdigit():
            versions[int(version)] = '/'.join([model_base_path, version])
    if version is None:
        version = max(versions.keys())
    return versions[int(version)]


def fetch_tensors_spark_schema(tensor_mappings):
    """
    Generate spark schema from tensor mappings.

    We skip the first dimension intentionally, so a tensor would fit in a spark dataframe.
    """

    def _tf_shape_as_array(shape, elem_type):
        """
        Generate spark Array schema from tensor shape.

        We use the tensor dtype map to the element spark sql data type.
        """
        if not shape:
            return elem_type()
        else:
            return T.ArrayType(_tf_shape_as_array(shape[1:], elem_type))

    return T.StructType([
        T.StructField(name,
                      _tf_shape_as_array(tensor.shape.as_list()[1:],
                                         TF_TO_SPARK_TYPE_MAPPINGS[tensor.dtype]))
        for name, tensor in tensor_mappings.items()
    ])


def _get_model_full_path(model_base_path, version=None):
    """
    Get the full path for a model.

    If version arg is not specified, we return the full path for the latest version.
    If version arg is specified, we verify that the version actually exists.
    """
    versions = {}
    DIGIT_PATTERN = '[0-9]*'
    model_base_path = model_base_path.rstrip('/')
    for path in tf.gfile.Glob(os.path.join(model_base_path, DIGIT_PATTERN)):
        _, v = path.rsplit('/', 1)
        versions[int(v)] = path

    if version is None:
        version = max(versions.keys())
    else:
        if not str(version).isdigit():
            raise Exception('Only int/long version is allowed')
        version = int(version)

    if version not in versions:
        raise Exception('Given version %s not found' % version)

    return versions[version]


def load_model(model_base_path=None, model_version=None, model_full_path=None, signature_def_key=None):
    """
    Load the model_version/latest savedModel from model_base_path.

    We use the default signature_def_key 'default_serving' by default.
    """
    assert model_base_path or model_full_path, "You have to specify either model_base_path or model_full_path"
    signature_def_key = signature_def_key or tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    model_full_path = model_full_path or _get_model_full_path(
        model_base_path, model_version)
    return tf.contrib.predictor.from_saved_model(model_full_path, signature_def_key)


def _unroll(a):
    """
    Flatten an Arrow-generated array structure

    Example:
    ```
    In [8]: import pyarrow as pa
    In [9]: df.iloc[0].to_dict()['a']
    Out [9]: [[1], [2]]

    In [10]: pa.Table.from_pandas(df).to_pandas().iloc[0].to_dict()['a']
    Out [10]: array([array([1]), array([2])], dtype=object)

    In [11]: pa.Table.from_pandas(df).to_pandas().apply(_unroll).iloc[0].to_dict()['a']
    Out [11]: [[1], [2]]
    ```
    """
    if hasattr(a, '__iter__'):
        return [_unroll(i) for i in a]
    if hasattr(a, 'tolist'):
        return a.tolist()
    return a


def _batch_predict(model, pandas_df):
    """
    Make predictions on a batch of input tensors.
    """
    def value_to_list(d): return {k: v.tolist() for k, v in d.iteritems()}
    return pd.DataFrame(value_to_list(model(pandas_df.to_dict('list'))))


def _record_predict(model, pandas_df):
    """
    Apply inference on the row basis.

    We need to expand the fist dimension before predict and squeeze the extra dimension after inference.
    """
    def expand_value_dim(d): return {k: [v] for k, v in d.iteritems()}

    def squeeze_value_dim(d): return {k: v[0] for k, v in d.iteritems()}
    return pd.DataFrame.from_records(
        squeeze_value_dim(model(expand_value_dim(record)))
        for record in pandas_df.to_dict('records'))


def _is_input_aligned(feed_tensors):
    """
    Check whether the input tensors have unfixed dimension other than the first one.
    """
    for name, tensor in feed_tensors.iteritems():
        if None in tensor.shape.as_list()[1:]:
            logging.warning('input: %s, has unfixed shape: %s' %
                            (name, tensor.shape))
            return False
    return True


def pandas_model_inference(model, pandas_df, output_columns):
    """
    TF inference on a panda dataframe with tf.contrib.predictor

    :param model: `tensorflow.contrib.predictor.Predictor`, an instance of a predictor class
    :param pandas_df: `pandas.DataFrame`, a pandas dataframe object
    :param output_columns: `List[str]`, list of strs, contains all the output columns.
    :return: `pandas.DataFrame`, a pandas dataframe contains only output_columns only.
    """
    output = pandas_df.apply(_unroll)
    feed_pdf = output[model.feed_tensors.keys()]
    if _is_input_aligned(model.feed_tensors):
        res = _batch_predict(model, feed_pdf)
    else:
        res = _record_predict(model, feed_pdf)

    res = pd.concat([res, output], axis=1)
    return res[output_columns]


class GraphDefPredictor(SavedModelPredictor):
    """
    Predictor from graph_def.

    Reusing the tf.contrib.predictor project.
    Details here https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/predictor
    """

    def __init__(self, graph_def, input_names, output_names, extra_op_names, config=None):
        with tf.Graph().as_default() as g:
            tf.import_graph_def(graph_def, name='')
            self._session = tf.Session(config=config)
            self._graph = g
            ops = tf.group([self._graph.get_operation_by_name(extra_op_name)
                            for extra_op_name in extra_op_names])
            self._session.run(ops)
            self._feed_tensors = {k: self._graph.get_tensor_by_name(v)
                                  for k, v in input_names.items()}
            self._fetch_tensors = {k: self._graph.get_tensor_by_name(v)
                                   for k, v in output_names.items()}

    @staticmethod
    def export_model(model):
        """
        Export a predictor into GraphDefPredictor required info
        """
        extra_ops = []
        legacy_collection_keys = ['table_initializer', 'legacy_init_op']
        all_collection_keys = model.graph.get_all_collection_keys()
        for key in legacy_collection_keys:
            if key in all_collection_keys:
                extra_ops.extend(
                    map(lambda o: o.name, model.graph.get_collection(key)))

        graph_def = tf.graph_util.convert_variables_to_constants(model.session,
                                                                 model.graph.as_graph_def(
                                                                     add_shapes=True),
                                                                 [fetch.op.name for fetch in model.fetch_tensors.values()] + extra_ops)
        feed_names = {key: value.name for key,
                      value in model.feed_tensors.iteritems()}
        fetch_names = {key: value.name for key,
                       value in model.fetch_tensors.iteritems()}

        return graph_def, feed_names, fetch_names, extra_ops
