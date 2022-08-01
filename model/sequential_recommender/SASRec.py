"""
Paper: Self-Attentive Sequential Recommendation
Author: Wang-Cheng Kang, and Julian McAuley
Reference: https://github.com/kang205/SASRec
@author: Zhongchuan Sun
"""
import tensorflow as tf
import numpy as np
from model.AbstractRecommender import SeqAbstractRecommender
from util import DataIterator
from util.tool import csr_to_user_dict_bytime

from util import inner_product
from util import batch_randint_choice
from util import pad_sequences


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):#tf.variable_scope相当于一个变量管理器，指明变量的作用域
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)#mean表示均值，variance表示方差，x,维度,keepdim表示是否保持维度，x上除去axes所指定的纬度的剩余纬度组成的各个子元素看做个体，个体中的每个位置的值看做个体的不同位置属性，然后求所有个体在每种位置属性上的均值和方差
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta    #对应论文LayerNorm=α*（x-μ）/根号下方差的平方+epsion+beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
      tf.variable() 和tf.get_variable()有不同的创建变量的方式：tf.Variable() 每次都会新建变量。如果希望重用（共享）一些变量，就需要用到了get_variable()，它会去搜索变量名，有就直接用，没有再新建。
    ```
    '''

    #embedding层
    #get_variable()希望重新用一些共享变量
    with tf.compat.v1.variable_scope(scope, reuse=reuse):#tf.variable() 和tf.get_variable()搭配使用在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
        lookup_table = tf.compat.v1.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.keras.regularizers.l2(l2_reg))
        if zero_pad:#布尔值，如果true第一行所有（id 0）的所有值都是常数0
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn. embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table,
    else:
        return outputs


def multihead_attention(queries,#定义多线程的注意，返回一个形状为三维张量
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False):

    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
#self-attention层
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]#返回一个元祖as_list用list形式
        tf.compat.v1.enable_eager_execution()
        # Linear projections线性投影
        # Q = tf.layers.dense(queries, num _units, activation=tf.nn.relu) # (N, T_q, C)(输入queries，输出维度num_units
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.compat.v1.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        #layers.dense 表示输入该网络的数据：queries,num_units:输出维度的大小，改变Input的最后一维
        

        K = tf.compat.v1.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.compat.v1.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h=num_heads) #线性投影将输入（embedding+位置）E_hat
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)划分拼接
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h) 把Q/K/V划分成num_heads=8份在第2维度

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k) 把k转置 (h*N, c/h, T/k) 第0维 公式（2）的（QK转置)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)#QK转置/根号d

        # Key Masking 秘钥屏蔽
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)/ tf.sign符号函数大于0返回1小于0返回-1，=0返回0 ，对Key tensor在维度-1即最后一维求和
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k) #tf.tillte在同一纬度上复制，input=key_mask,前复制num_heads次后复制1次
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)根据Key_masks和0是否相等返回padding或outputs

        # Causality = Future blinding 对未来信息的掩盖
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k) 创建一个和输入参数（tensor）维度一样，元素都为1的张量
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k) #用于线性计算
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking 查询屏蔽
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


class SASRec(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(SASRec, self).__init__(dataset, conf)
        train_matrix, time_matrix = dataset.train_matrix, dataset.time_matrix
        self.dataset = dataset

        self.users_num, self.items_num = train_matrix.shape

        self.lr = conf["lr"]
        self.l2_emb = conf["l2_emb"]
        self.hidden_units = conf["hidden_units"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]
        self.dropout_rate = conf["dropout_rate"]
        self.max_len = conf["max_len"]
        self.num_blocks = conf["num_blocks"]
        self.num_heads = conf["num_heads"]

        self.user_pos_train = csr_to_user_dict_bytime(time_matrix, train_matrix)

        self.sess = sess

    def _create_variable(self):
        # self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seq_ph = tf.compat.v1.placeholder(tf.int32, [None, self.max_len], name="item_seq")
        self.item_pos_ph = tf.compat.v1.placeholder(tf.int32, [None, self.max_len], name="item_pos")
        self.item_neg_ph = tf.compat.v1.placeholder(tf.int32, [None, self.max_len], name="item_neg")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="training_flag")

        l2_regularizer = tf.keras.regularizers.l2(self.l2_emb)
        item_embeddings = tf.compat.v1.get_variable('item_embeddings', dtype=tf.float32,
                                          shape=[self.items_num, self.hidden_units],
                                          regularizer=l2_regularizer)

        zero_pad = tf.zeros([1, self.hidden_units], name="padding")
        item_embeddings = tf.concat([item_embeddings, zero_pad], axis=0)
        self.item_embeddings = item_embeddings * (self.hidden_units ** 0.5)

        self.position_embeddings = tf.compat.v1.get_variable('position_embeddings', dtype=tf.float32,
                                                   shape=[self.max_len, self.hidden_units],
                                                   regularizer=l2_regularizer)

    def build_graph(self):  #生成seq_embed并加上pos_embed
        self._create_variable()
        reuse = None
        with tf.compat.v1.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq = tf.nn.embedding_lookup(self.item_embeddings, self.item_seq_ph) #选取一个张量里索引对应的元素
            item_emb_table = self.item_embeddings

            # Positional Encoding
            position = tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq_ph)[1]), 0),
                               [tf.shape(self.item_seq_ph)[0], 1])
            t = tf.nn.embedding_lookup(self.position_embeddings, position)
            # pos_emb_table = self.position_embeddings

            self.seq += t

            # Dropout
            self.seq = tf.compat.v1.layers.dropout(self.seq,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            mask = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.item_seq_ph, self.items_num)), -1)
            self.seq *= mask

            # Build blocks

            for i in range(self.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq),
                                           num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate,
                                           is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)  # (b, l, d)
            last_emb = self.seq[:, -1, :]  # (b, d), the embedding of last item of each session

        pos = tf.reshape(self.item_pos_ph, [tf.shape(self.item_seq_ph)[0] * self.max_len])  # (b*l,)
        neg = tf.reshape(self.item_neg_ph, [tf.shape(self.item_seq_ph)[0] * self.max_len])  # (b*l,)
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)  # (b*l, d)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)  # (b*l, d)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.item_seq_ph)[0] * self.max_len, self.hidden_units])  # (b*l, d)

        # prediction layer
        self.pos_logits = inner_product(pos_emb, seq_emb)  # (b*l,)
        self.neg_logits = inner_product(neg_emb, seq_emb)  # (b*l,)

        # ignore padding items (self.items_num)
        is_target = tf.reshape(tf.compat.v1.to_float(tf.not_equal(pos, self.items_num)),
                               [tf.shape(self.item_seq_ph)[0] * self.max_len])

        pos_loss = -tf.compat.v1.log(tf.sigmoid(self.pos_logits) + 1e-24) * is_target
        neg_loss = -tf.compat.v1.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * is_target
        self.loss = tf.reduce_sum(pos_loss + neg_loss) / tf.reduce_sum(is_target)

        try:
            reg_losses = tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = self.loss + reg_losses
        except:
            pass

        self.train_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta2=0.98).minimize(self.loss)

        # for predication/test
        items_embeddings = item_emb_table[:-1]  # remove the padding item
        self.all_logits = tf.matmul(last_emb, items_embeddings, transpose_b=True)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())

        for epoch in range(self.epochs):
            item_seq_list, item_pos_list, item_neg_list = self.get_train_data()
            data = DataIterator(item_seq_list, item_pos_list, item_neg_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_item_seq, bat_item_pos, bat_item_neg in data:
                feed = {self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg,
                        self.is_training: True}

                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def get_train_data(self):
        item_seq_list, item_pos_list, item_neg_list = [], [], []
        all_users = DataIterator(list(self.user_pos_train.keys()), batch_size=1024, shuffle=False)
        for bat_users in all_users:
            bat_seq = [self.user_pos_train[u][:-1] for u in bat_users]
            bat_pos = [self.user_pos_train[u][1:] for u in bat_users]
            n_neg_items = [len(pos) for pos in bat_pos]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion)
            for i in range(len(bat_neg)):
                if type(bat_neg[i]) == int:
                    bat_neg[i] = [bat_neg[i]]
            # padding

            bat_seq = pad_sequences(bat_seq, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')
            bat_pos = pad_sequences(bat_pos, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')
            bat_neg = pad_sequences(bat_neg, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')

            item_seq_list.extend(bat_seq)
            item_pos_list.extend(bat_pos)
            item_neg_list.extend(bat_neg)
        return item_seq_list, item_pos_list, item_neg_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        users = DataIterator(users, batch_size=512, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            bat_seq = [self.user_pos_train[u] for u in bat_user]
            bat_seq = pad_sequences(bat_seq, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')
            feed = {self.item_seq_ph: bat_seq,
                    self.is_training: False}
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings)
        all_ratings = np.array(all_ratings, dtype=np.float32)
        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]
        return all_ratings
