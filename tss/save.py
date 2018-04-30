import logging
import os
import time

import tensorflow as tf


def create_tmp_path(output_dir):
    tmp_export_folder = 'tmp-' + str(int(time.time()))
    tmp_path = os.path.join(output_dir, tmp_export_folder)
    if tf.gfile.Exists(tmp_path):
        raise RuntimeError('Failed to obtain tmp export directory')
    return tmp_path


def export_saved_model(inputs, outputs, output_dir, model_version, graph, sess=None, method_name='tensorflow/serving/classify'):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    sess = sess or tf.Session()

    output_path = os.path.join(
        tf.compat.as_bytes(output_dir),
        tf.compat.as_bytes(str(model_version)))

    tmp_path = create_tmp_path(output_dir)

    with graph.as_default():
        with sess.as_default():
            logging.info('Exporting trained model to %s' % output_path)
            builder = tf.saved_model.builder.SavedModelBuilder(tmp_path)

            signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        k: tf.saved_model.utils.build_tensor_info(v)
                        for k, v in inputs.items()
                    },
                    outputs={
                        k: tf.saved_model.utils.build_tensor_info(v)
                        for k, v in outputs.items()
                    },
                    method_name=method_name)
               )


            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        signature,
                })
            builder.save()

    try:
        tf.gfile.Rename(tmp_path, output_path)
    except:
        raise Exception('Model version already exists: %s'%output_path)

    if tf.gfile.Exists(tmp_path):
        tf.gfile.DeleteRecursively(tmp_path)
    logging.info('Successfully exported model to %s' % output_dir)
