import tensorflow as tf
from tensorflow.python.framework import ops as _ops
from neuralgym.utils import get_sess


def collection_to_dict(collection):
    """Utility function to construct collection dict with names."""
    return {v.name[:v.name.rfind(':')]: v for v in collection}


def scalar_summary(name, value, sess=None, summary_writer=None, step=None):
    """Add scalar summary.

    In addition to summary tf.Tensor and tf.Variable, this function supports
    summary of constant values by creating placeholder.

    Example usage:

    >>> scalar_summary('lr', lr)

    :param name: name of summary variable
    :param value: numpy or tensorflow tensor
    :param summary_writer: if summary writer is provided, write to summary
        instantly
    :param step: if summary writer is provided, write to summary with step
    :return: None
    """
    def is_tensor_or_var(value):
        return isinstance(value, tf.Tensor) or isinstance(value, tf.Variable)

    # get scope name
    sname = tf.get_variable_scope().name
    fname = name if not sname else sname+'/'+name
    # construct summary dict
    collection = collection_to_dict(
        _ops.get_collection(_ops.GraphKeys.SUMMARIES))
    # tensorflow tensor
    if fname in collection:
        if not is_tensor_or_var(value):
            ph_collection = collection_to_dict(
                tf.get_collection('SUMMARY_PLACEHOLDERS'))
            op_collection = collection_to_dict(
                tf.get_collection('SUMMARY_OPS'))
            ph = ph_collection[fname+'_ph']
            op = op_collection[fname+'_op']
            sess = get_sess(sess)
            sess.run(op, feed_dict={ph: value})
        summary = collection[fname]
    else:
        if not is_tensor_or_var(value):
            print(
                'To write summary, create tensor for non-tensor value: '
                '%s_var.' % name)
            # create a summary variable
            value = tf.Variable(value, name=name+'_var')
            ph = tf.placeholder(value.dtype, name=name+'_ph')
            op = tf.assign(value, ph, name=name+'_op')
            tf.add_to_collection('SUMMARY_PLACEHOLDERS', ph)
            tf.add_to_collection('SUMMARY_OPS', op)
            # initialize variable
            sess = get_sess(sess)
            sess.run(tf.initialize_variables([value]))
        # new summary tensor
        with tf.device('/cpu:0'):
            summary = tf.summary.scalar(name, value)
    # write to summary
    if summary_writer is not None:
        assert step is not None, 'step be None when write to summary.'
        sess = get_sess(sess)
        summary_writer.add_summary(sess.run(summary), step)


def images_summary(images, name, max_outs, color_format='BGR'):
    """Summary images.

    **Note** that images should be scaled to [-1, 1] for 'RGB' or 'BGR',
    [0, 1] for 'GREY'.

    :param images: images tensor (in NHWC format)
    :param name: name of images summary
    :param max_outs: max_outputs for images summary
    :param color_format: 'BGR', 'RGB' or 'GREY'
    :return: None
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        if color_format == 'BGR':
            img = tf.clip_by_value(
                (tf.reverse(images, [-1])+1.)*127.5, 0., 255.)
        elif color_format == 'RGB':
            img = tf.clip_by_value((images+1.)*127.5, 0, 255)
        elif color_format == 'GREY':
            img = tf.clip_by_value(images*255., 0, 255)
        else:
            raise NotImplementedError("color format is not supported.")
        tf.summary.image(name, img, max_outputs=max_outs)


def gradients_summary(y, x, norm=tf.abs, name='gradients_y_wrt_x'):
    """Summary gradients w.r.t. x.

    Sum of norm of :math:`\\nabla_xy`.

    :param y: y
    :param x: w.r.t x
    :param norm: norm function, default is tf.abs
    :param name: name of gradients summary
    :return: None
    """
    grad = tf.reduce_sum(norm(tf.gradients(y, x)))
    scalar_summary(name, grad)
