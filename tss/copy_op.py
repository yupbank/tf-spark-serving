import tensorflow as tf
from copy import deepcopy


def copy_tensor_to_graph(tensor, to_graph, cached_tensors, scope=''):
    if scope != '':
        new_name = scope + '/' + tensor.name
    else:
        new_name = tensor.name

    if new_name in cached_tensors:
        return cached_tensors[new_name]

    op = tensor.op
    new_op = copy_op_to_graph(op, to_graph, cached_tensors, scope)
    output_index = op.outputs.index(tensor)
    new_tensor = new_op.outputs[output_index]
    cached_tensors[tensor.name] = new_tensor
    return new_tensor


def copy_op_to_graph(op, to_graph, cached_tensors, scope=''):
    if scope != '':
        new_name = scope + '/' + op.name
    else:
        new_name = op.name

    try:
        already_present = to_graph.as_graph_element(
            new_name, allow_tensor=True, allow_operation=True)
        return already_present
    except:
        pass

    if op._original_op is not None:
        new_original_op = copy_op_to_graph(op._original_op, to_graph, cached_tensors,
                                           scope)
    else:
        new_original_op = None

    new_control_inputs = [
        copy_op_to_graph(x, to_graph, cached_tensors, scope)
        for x in op.control_inputs
    ]

    #If it has inputs, call this function recursively on each.
    new_inputs = [
        copy_tensor_to_graph(x, to_graph, cached_tensors, scope) for x in op.inputs
    ]

    #Make a new node_def based on that of the original.
    #An instance of tensorflow.core.framework.node_def_pb2.NodeDef, it
    #stores String-based info such as name, device and type of the op.
    #Unique to every Operation instance.
    new_node_def = deepcopy(op.node_def)
    #Change the name
    new_node_def.name = new_name

    #Copy the other inputs needed for initialization
    output_types = op._output_types[:]
    input_types = op._input_types[:]

    #Make a copy of the op_def too.
    #Its unique to every _type_ of Operation.
    op_def = deepcopy(op.op_def)

    #Initialize a new Operation instance
    new_op = tf.Operation(new_node_def, to_graph, new_inputs, output_types,
                          new_control_inputs, input_types, new_original_op,
                          op_def)
    #Use Graph's hidden methods to add the op
    to_graph._record_op_seen_by_control_dependencies(new_op)
    # pylint: disable=protected-access
    for device_function in to_graph._device_functions_outer_to_inner:
      new_op._set_device(device_function(new_op))
    # pylint: enable=protected-access

    return new_op


def eval_graph(input_mappings, outputs, to_graph, scope=''):
    if scope:
        cached_tensors = {'/'.join([scope, k.name]): v for k, v in input_mappings.items()}
    else:
        cached_tensors = {k.name: v for k, v in input_mappings.items()}

    return [
        copy_tensor_to_graph(tensor, to_graph, cached_tensors, scope)
        for tensor in outputs]


if __name__ == "__main__":
    with tf.Graph().as_default() as g1:
        a = tf.placeholder(tf.float32, [None], name='a')
        b = tf.placeholder(tf.float32, [None], name='b')
        z = tf.no_op()
        with tf.control_dependencies([z]):
            d = a + b + b + a + a

    with tf.Graph().as_default() as g2:
        x = tf.placeholder(tf.float32, [None], name='x')
        y = tf.placeholder(tf.float32, [None], name='y')
        l = tf.placeholder(tf.float32, [None], name='a')
        z = eval_graph({a: x, b: y}, [d], g2, '')
        sess = tf.Session()
        print(sess.run(z, {x: [0.1, 0.2], y: [0.2, 0.3]}))
        print(sess.run(z, {x: [0.1, 0.2], y: [0.2, 0.3], l: [0.2, 0.4]}))
        z = eval_graph({a: x, b: y}, [d], g2, 'm')
        print(sess.run(z, {x: [0.1, 0.2], y: [0.2, 0.3]}))
        print(sess.run(z, {x: [0.1, 0.2], y: [0.2, 0.3], l: [0.2, 0.4]}))
