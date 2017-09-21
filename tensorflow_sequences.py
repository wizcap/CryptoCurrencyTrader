import tensorflow as tf


def DataIterator():
    def __init__(self, input_data, target_data):
        self.size = len(target_data)
        self.epochs = 0
        self.input_data = input_data
        self.target_data = target_data

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
        input_data = self.input_data[self.cursor:self.cursor+n-1]
        target_data = self.target_data[self.cursor:self.cursor+n-1]
        self.cursor += n
        return input_data, target_data


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def train_graph(graph, batch_size = 256, num_epochs = 10, iterator=DataIterator):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tr = iterator(train)
        te = iterator(test)

        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_

                te_losses.append(accuracy / step)
                step, accuracy = 0,0
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])

    return tr_losses, te_losses


def build_seq2seq_graph(
    vocab_size = 256,
    state_size = 64,
    batch_size = 256,
    num_classes = 2):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, None])
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(1.0)

    y_ = tf.tile(tf.expand_dims(y, 1), [1, tf.shape(x)[1]])

    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size],
                                 initializer=tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen,
                                                 initial_state=init_state)

    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y_, [-1])

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(rnn_outputs, W) + b

    preds = tf.nn.softmax(logits)

    correct = tf.cast(tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y_reshaped),tf.int32)

    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(seqlen, tf.float32))

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped)
    loss = tf.reduce_sum(loss)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }
