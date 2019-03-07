import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
import logging

from utils import load_model, fetch_tensors_spark_schema, GraphDefPredictor, pandas_model_inference


def load_meta_graph(model_path, tags, graph, signature_def_key=None):
    saved_model = reader.read_saved_model(model_path)
    the_meta_graph = None
    for meta_graph_def in saved_model.meta_graphs:
        if sorted(meta_graph_def.meta_info_def.tags) == sorted(tags):
            the_meta_graph = meta_graph_def
    signature_def_key = signature_def_key or tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    try:
        signature_def = signature_def_utils.get_signature_def_by_key(
            the_meta_graph,
            signature_def_key)
    except ValueError as ex:
        try:
            formatted_key = 'default_input_alternative:{}'.format(
                signature_def_key)
            signature_def = signature_def_utils.get_signature_def_by_key(
                the_meta_graph, formatted_key)
        except ValueError:
            raise ValueError(
                'Got signature_def_key "{}". Available signatures are {}. '
                'Original error:\n{}'.format(
                    signature_def_key, list(the_meta_graph.signature_def), ex)
            )
    input_names = {k: v.name for k, v in signature_def.inputs.items()}
    output_names = {k: v.name for k, v in signature_def.outputs.items()}
    feed_tensors = {k: graph.get_tensor_by_name(v)
                    for k, v in input_names.items()}
    fetch_tensors = {k: graph.get_tensor_by_name(v)
                     for k, v in output_names.items()}
    return feed_tensors, fetch_tensors


def get_model(model_base_path, version=None):
    versions = {}
    model_base_path = model_base_path.rstrip('/')
    for version in tf.gfile.ListDirectory(model_base_path):
        version = version.rstrip('/')
        if version.isdigit():
            versions[int(version)] = '/'.join([model_base_path, version])
    if version is None:
        version = max(versions.keys())
    logging.info('Choose model_path %s with version %s' %
                 (model_base_path, version))
    return versions[int(version)]


def load_model(model_base_path, model_version):
    model_path = get_model(model_base_path, model_version)
    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        tags = [tf.saved_model.tag_constants.SERVING]
        tf.saved_model.loader.load(sess, tags, model_path)
        feed_tensors, fetch_tensors = load_meta_graph(model_path, tags, graph)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                        graph.as_graph_def(
                                                                            add_shapes=True),
                                                                        [fetch.op.name for fetch in fetch_tensors.values()])
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(output_graph_def, name='')
        return graph, \
            {name: graph.get_tensor_by_name(feed.name) for name, feed in feed_tensors.items()},\
            {name: graph.get_tensor_by_name(fetch.name)
             for name, fetch in fetch_tensors.items()}


def rename_by_mapping(df, tensor_mapping, reverse=False):
    for name, tensor in tensor_mapping.items():
        renames = name, tensor.op.name
        if reverse:
            renames = reversed(renames)
        df = df.withColumnRenamed(*renames)
    return df


def tf_serving_with_pandas(df, model_base_path, model_version=None):
    from pyspark.sql.functions import pandas_udf, PandasUDFType


def tf_serving_with_broadcasted_model(df, model_base_path=None, model_version=None, model_full_path=None, signature_def_key=None):
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    model = load_model(model_base_path, model_version,
                       model_full_path, signature_def_key)
    tf_output_schema = fetch_tensors_spark_schema(model.fetch_tensors)
    output_schema = T.StructType(df.schema.fields + tf_output_schema.fields)

    graph_def, feed_names, fetch_names, extra_ops = GraphDefPredictor.export_model(
        model)
    graph_def_serialized_bc = df.rdd.context.broadcast(graph_def)

    def func(pandas_df):
        """
        Batch inference on a panda dataframe
        """
        predictor_model = GraphDefPredictor(
            graph_def_serialized_bc.value, feed_names, fetch_names, extra_ops)
        return pandas_model_inference(predictor_model, pandas_df, output_schema.fieldNames())

    inference = F.pandas_udf(func, output_schema, F.PandasUDFType.GROUPED_MAP)
    return df.groupby(F.spark_partition_id()).apply(inference)


def tf_serving_with_dataframe(df, model_base_path, model_version=None):
    """

    :param df: spark dataframe, batch input for the model
    :param model_base_path: str, tensorflow saved Model model base path
    :param model_version: int, tensorflow saved Model model version, default None
    :return: spark dataframe, with predicted result.
    """
    import tensorframes as tfs
    g, feed_tensors, fetch_tensors = load_model(model_base_path, model_version)
    with g.as_default():
        df = rename_by_mapping(df, feed_tensors)
        df = tfs.analyze(df)
        df = tfs.map_blocks(fetch_tensors.values(), df)
        df = rename_by_mapping(df, feed_tensors, reverse=True)
        return rename_by_mapping(df, fetch_tensors, reverse=True)
