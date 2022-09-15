
import numpy as np
import scipy.sparse as sp
from model.AbstractRecommender import SeqAbstractRecommender
from util import DataIterator, timer
from util.tool import csr_to_user_dict_bytime, csr_to_time_dict, csr_to_user_dict
import tensorflow as tf
import math
from util.cython.random_choice import batch_randint_choice
# from util import batch_randint_choice
from util import l2_loss

class GCTN(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(GCTN, self).__init__(dataset, conf)
        train_matrix, time_matrix = dataset.train_matrix, dataset.time_matrix
        test_matrix = dataset.test_matrix
        self.test_time_matrix = dataset.time_test_matrix
        self.dataset = dataset
        self.users_num, self.items_num = dataset.train_matrix.shape
        self.cat_num = len(np.unique(dataset.cat_matrix.data))
        self.lr = conf["lr"]
        self.l2_reg = conf["l2_reg"]
        self.l2_regW = conf["l2_regW"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]
        self.dropout_rate = conf["dropout_rate"]
        self.hidden_units = conf["hidden_units"]
        self.num_blocks = conf["num_blocks"]
        self.num_heads = conf["num_heads"]
        self.seq_L = conf["seq_L"]
        self.seq_T = conf["seq_T"]
        self.neg_samples = conf["neg_samples"]
        self.user_pos_train = csr_to_user_dict_bytime(time_matrix, train_matrix)
        self.user_pos_time = csr_to_time_dict(time_matrix)
        self.user_pos_test = csr_to_user_dict(test_matrix)
        # GCN's hyperparameters
        self.n_layers = conf['n_layers']
        self.norm_adj = self.create_adj_mat(conf['adj_type'])
        self.user_test_time = {}
        for user_id in range(self.users_num):
            seq_timeone = self.test_time_matrix[user_id, self.user_pos_test[user_id][0]]
            seq_times = self.user_pos_time[user_id]
            content_time = list()
            size = len(seq_times)
            for index in range(min([self.seq_L, size])):
                deltatime_now = abs(seq_times[size - index - 1] - seq_timeone) / (3600 * 24)
                if deltatime_now == 0:
                    deltatime_now = 1 / (3600 * 24)
                content_time.append(-math.log(deltatime_now))
            if (size < self.seq_L):
                content_time = content_time + [self.items_num for _ in range(self.seq_L - len(content_time))]
            self.user_test_time[user_id] = content_time
        self.sess = sess

    @timer
    def create_adj_mat(self, adj_type):
        user_list, item_list, category_list = self.dataset.get_train_interactionssecond()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        category_np = np.array(category_list, dtype=np.float32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.users_num + self.items_num + self.cat_num

        tmp_adjone = sp.csr_matrix((ratings, (user_np, item_np + self.users_num + self.cat_num)), shape=(n_nodes, n_nodes))
        tmp_adjtwo = sp.csr_matrix((ratings, (user_np, category_np + self.users_num)), shape=(n_nodes, n_nodes))
        tmp_adjtwo[tmp_adjtwo >= 1] = 1
        a = sp.csr_matrix((ratings, (item_np + self.users_num, category_np + self.users_num)), shape=(n_nodes, n_nodes))
        a[a >= 1] = 1
        tmp_adj_second = a + a.T
        tmp_adj = tmp_adjone + tmp_adjtwo
        adj_mat = tmp_adj + tmp_adj.T + tmp_adj_second

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix

    def _create_gcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.embeddings["user_embeddings"], self.embeddings["category_embeddings"], self.embeddings["item_embeddings"]],axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(adj_mat, ego_embeddings,name="sparse_dense")

            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, c_g_embeddings, i_g_embeddings = tf.split(all_embeddings,[self.users_num, self.cat_num, self.items_num], 0)
        return u_g_embeddings, c_g_embeddings, i_g_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def normalize(self,
                  inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention",
                            with_qk=False):
        with tf.compat.v1.variable_scope(scope, reuse=True):
            if num_units is None:
                self.num_units = queries.get_shape().as_list[-1]
            Q = tf.matmul(queries, self.weights['attention_Q'])
            K = tf.matmul(keys, self.weights['attention_K'])
            V = tf.matmul(keys, self.weights['attention_V'])

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking 秘钥屏蔽
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

            # Activation
            outputs = tf.nn.softmax(outputs)

            # Query Masking 查询屏蔽
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
            query_masks = tf.tile(query_masks, [num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks

            # Dropouts
            outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate,
                                                  training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

            # Residual connection
            outputs += queries

        if with_qk:
            return Q, K
        else:
            return outputs

    def feedforward(self,
                    input,
                    scope="multihead_attention",
                    dropout_rate=0.5,
                    is_training=True,
                    reuse=None):

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Inner layer
            self.weights["b1_"] = tf.expand_dims(self.weights["b1"], axis=0)
            feedforward_output = tf.matmul(input, self.weights["feedforward_W"]) + tf.expand_dims(self.weights["b1_"], axis=0)
            outputs1 = tf.nn.relu(feedforward_output)
            outputs = tf.compat.v1.layers.dropout(outputs1, rate=dropout_rate,
                                                  training=tf.convert_to_tensor(is_training))

            # Readout layer
            self.weights["b2_"] = tf.expand_dims(self.weights["b2"], axis=0)
            feedforward_output = tf.matmul(outputs, self.weights["feedforward_b"]) + tf.expand_dims(self.weights["b2_"], axis=0)
            outputs = tf.compat.v1.layers.dropout(feedforward_output, rate=dropout_rate,
                                                  training=tf.convert_to_tensor(is_training))

            # Residual connection
            outputs += input

        return outputs

    def _create_placeholder(self):
        self.user_ph = tf.compat.v1.placeholder(tf.int32, [None], name="user")
        self.item_seq_ph = tf.compat.v1.placeholder(tf.int32, [None, self.seq_L], name="item_seq")
        self.item_pos_ph = tf.compat.v1.placeholder(tf.int32, [None, self.seq_T], name="item_pos")
        self.item_neg_ph = tf.compat.v1.placeholder(tf.int32, [None, self.neg_samples], name="item_neg")
        self.timenow_ph = tf.compat.v1.placeholder(tf.float32, [None, self.seq_L], name="time_now")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="training_flag")

    def _create_variable(self):
        self.embeddings = dict()
        self.weights = dict()

        Weight_initializer = tf.initializers.variance_scaling(scale=2.0, mode='fan_in')
        self.weights['weight_time'] = tf.Variable(Weight_initializer([self.hidden_units, self.hidden_units]))
        self.weights['attention_Q'] = tf.Variable(Weight_initializer([self.hidden_units, self.hidden_units]))
        self.weights['attention_K'] = tf.Variable(Weight_initializer([self.hidden_units, self.hidden_units]))
        self.weights['attention_V'] = tf.Variable(Weight_initializer([self.hidden_units, self.hidden_units]))
        self.weights['feedforward_W'] = tf.Variable(Weight_initializer([self.hidden_units, self.hidden_units]))
        self.weights['feedforward_b'] = tf.Variable(Weight_initializer([self.hidden_units, self.hidden_units]))
        self.weights['b1'] = tf.Variable(Weight_initializer([self.hidden_units]), dtype=tf.float32)
        self.weights['b2'] = tf.Variable(Weight_initializer([self.hidden_units]), dtype=tf.float32)

        embeding_initializer = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.01)

        # SASRec embedding
        seq_item_embeddings = tf.Variable(embeding_initializer([self.items_num, self.hidden_units]), dtype=tf.float32)
        zero_pad = tf.zeros([1, self.hidden_units], name="padding")
        seq_item_embeddings = tf.concat([seq_item_embeddings, zero_pad], axis=0)
        self.seq_item_embeddings = seq_item_embeddings * (self.hidden_units ** 0.5)

        # GCN embedding
        user_embeddings = tf.Variable(embeding_initializer([self.users_num, self.hidden_units]), dtype=tf.float32)
        self.embeddings.setdefault("user_embeddings", user_embeddings)
        # predication embedding
        item_embeddings = tf.Variable(embeding_initializer([self.items_num, self.hidden_units]), dtype=tf.float32)
        self.embeddings.setdefault("item_embeddings", item_embeddings)
        category_embeddings = tf.Variable(embeding_initializer([self.cat_num, self.hidden_units]), dtype=tf.float32)
        self.embeddings.setdefault("category_embeddings", category_embeddings)
        # GCN embedding
        self.user_embeddings, self.category_embeddings, self.item_embeddings = self._create_gcn_embed()

    def _create_inference(self):

        with tf.compat.v1.variable_scope("text", reuse=True):
            # user encoding
            self.user_embs = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # (b, d)
            user_embs = tf.expand_dims(self.user_embs, axis=1)

            self.item_embs = tf.nn.embedding_lookup(self.seq_item_embeddings, self.item_seq_ph)

            mask = tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(self.item_seq_ph, self.items_num)), -1)

            # Positional Encoding
            weight_mlp = tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq_ph)[1]), 0),
                                 [tf.shape(self.item_seq_ph)[0], 1])
            MLP = tf.nn.embedding_lookup(self.weights['weight_time'], weight_mlp)  # b,L,d

            relative_times = tf.tile(tf.expand_dims(self.timenow_ph, -1), tf.stack([1, 1, self.hidden_units]))  # b,L,d

            relative_position_embeddings = tf.multiply(MLP, relative_times)

            final_seq_embeddings = self.item_embs + user_embs + relative_position_embeddings

            # final_seq_embeddings *= mask

            self.item_embs = tf.compat.v1.layers.dropout(final_seq_embeddings,
                                                         rate=self.dropout_rate,
                                                         training=tf.convert_to_tensor(self.is_training))
            self.item_embs *= mask
            # Build blocks
            for i in range(self.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.item_embs = self.multihead_attention(queries=self.normalize(self.item_embs),
                                                              keys=self.item_embs,
                                                              num_units=self.hidden_units,
                                                              num_heads=self.num_heads,
                                                              dropout_rate=self.dropout_rate,
                                                              is_training=self.is_training,
                                                              causality=True,
                                                              scope="self_attention")
                    # Feed forward
                    self.item_embs = self.feedforward(self.normalize(self.item_embs),
                                                      dropout_rate=self.dropout_rate,
                                                      is_training=self.is_training)
                    self.item_embs *= mask

            self.item_embs = self.normalize(self.item_embs)# (b, l, d)
            self.final_user_embeddings = tf.expand_dims(self.item_embs[:, -1, :], axis=1) + user_embs

        self.item_embedding_pos = tf.nn.embedding_lookup(self.item_embeddings, self.item_pos_ph)  # b,T,d
        self.item_embedding_neg = tf.nn.embedding_lookup(self.item_embeddings, self.item_neg_ph)  # b,N,d
        tar_item_embs = tf.concat([self.item_embedding_pos, self.item_embedding_neg], axis=1)

        logits = tf.squeeze(tf.matmul(self.final_user_embeddings, tar_item_embs, transpose_b=True), axis=1)

        self.pos_logits, self.neg_logits = tf.split(logits, [self.seq_T, self.neg_samples], axis=1)

        # cross entropy loss
        pos_loss = tf.reduce_sum(-tf.compat.v1.log(tf.sigmoid(self.pos_logits) + 1e-24))
        neg_loss = tf.reduce_sum(-tf.compat.v1.log(1 - tf.sigmoid(self.neg_logits) + 1e-24))
        loss = pos_loss + neg_loss

        self.L2_weight = tf.reduce_sum(tf.square(self.weights["attention_Q"])) + \
                         tf.reduce_sum(tf.square(self.weights["attention_K"])) + \
                         tf.reduce_sum(tf.square(self.weights["attention_V"])) + \
                         tf.reduce_sum(tf.square(self.weights["weight_time"])) + \
                         tf.reduce_sum(tf.square(self.weights["feedforward_W"])) + \
                         tf.reduce_sum(tf.square(self.weights["feedforward_b"])) + \
                         tf.reduce_sum(tf.square(self.weights["b1"])) + \
                         tf.reduce_sum(tf.square(self.weights["b2"]))  # mlp

        # 针对GCN随机初始化的矩阵， lookup出对应的user和item向量，写入正则化，以便正则化新的矩阵
        Tpos = tf.nn.embedding_lookup(self.embeddings["item_embeddings"], self.item_pos_ph)
        Tneg = tf.nn.embedding_lookup(self.embeddings["item_embeddings"], self.item_neg_ph)
        user = tf.nn.embedding_lookup(self.embeddings["user_embeddings"], self.user_ph)

        self.Loss_0 = loss + self.l2_reg * l2_loss(self.item_embs, Tpos, Tneg, user, relative_position_embeddings) + self.l2_regW * self.L2_weight

        # for predication/test
        self.all_logits = tf.matmul(tf.squeeze(self.final_user_embeddings, axis=1), self.item_embeddings, transpose_b=True)

    def _create_optimizer(self):
        self.train_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.Loss_0)

    def build_graph(self):
        self._create_placeholder()
        self._create_variable()
        self._create_inference()
        self._create_optimizer()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        users_list, item_seq_list, item_pos_list, time_list = self._generate_sequences()
        for epoch in range(self.epochs):
            item_neg_list = self._sample_negative(users_list)
            data = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list, time_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg, bat_time in data:
                feed = {self.user_ph: bat_user,
                        self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg,
                        self.timenow_ph: bat_time,
                        self.is_training: True}

                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _generate_sequences(self):
        self.user_test_seq = {}
        user_list, item_seq_list, item_pos_list, time_list = [], [], [], []
        userid_set = np.unique(list(self.user_pos_train.keys()))

        for user_id in userid_set:
            seq_items = self.user_pos_train[user_id]
            seq_time = self.user_pos_time[user_id]
            for index_id in range(len(seq_items)):
                if index_id < self.seq_T: continue
                content_data = list()
                content_time = list()
                self.seq_timeone = seq_time[min([index_id + 1, len(seq_items) - 1])]
                for cindex in range(max([0, index_id - self.seq_L - self.seq_T + 1]), index_id - self.seq_T + 1):  # 根据序列L长度从一开始到结束，索引所在的位置到索引减去序列长度所指示的位置
                    content_data.append(seq_items[cindex])
                    deltatime_now = abs((seq_time[cindex] - self.seq_timeone)) / (3600 * 24)
                    if deltatime_now == 0:
                        deltatime_now = 1 / (3600 * 24)
                    content_time.append(-math.log(deltatime_now))
                if (len(content_data) < self.seq_L):
                    content_data = content_data + [self.items_num for _ in range(self.seq_L - len(content_data))]
                    content_time = content_time + [self.items_num for _ in range(self.seq_L - len(content_time))]# for_in相当于循环里的for i in,为什么是item_num
                user_list.append(user_id)
                time_list.append(content_time)
                item_seq_list.append(content_data)
                item_pos_list.append(seq_items[index_id - self.seq_T + 1:index_id + 1])

            user_id_seq = seq_items[-min([len(seq_items), self.seq_L]):]
            if (len(seq_items) < self.seq_L):
                user_id_seq = user_id_seq + [self.items_num for _ in range(self.seq_L - len(user_id_seq))]
            self.user_test_seq[user_id] = user_id_seq

        return user_list, item_seq_list, item_pos_list, time_list

    def _sample_negative(self, users_list):
        neg_items_list = []
        user_neg_items_dict = {}
        all_uni_user, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_uni_user, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            n_neg_items = [c * self.neg_samples for c in bat_counts]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion)
            for u, neg in zip(bat_users, bat_neg):
                user_neg_items_dict[u] = neg

        for u, c in zip(all_uni_user, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c, self.neg_samples])
            neg_items_list.extend(neg_items)
        return neg_items_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        users = DataIterator(users, batch_size=512, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            bat_seq = [self.user_test_seq[u] for u in bat_user]
            bat_seq_time = [self.user_test_time[u] for u in bat_user]
            feed = {self.user_ph: bat_user,
                    self.item_seq_ph: bat_seq,
                    self.timenow_ph: bat_seq_time,
                    self.is_training: False}
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings)
        all_ratings = np.array(all_ratings, dtype=np.float32)
        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]
        return all_ratings

