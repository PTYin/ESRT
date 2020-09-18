import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


from esrt.engine.base_model import BaseModel
from esrt.losses import single_nce_loss, pair_search_loss
from esrt.query_embedding import get_query_embedding
from tensorflow.python.framework import dtypes


class LSE(BaseModel):
    def __init__(self, dataset, params, forward_only=False):
        print("################LSE####################")
        self._dataset = dataset
        self.vocab_size = self._dataset.vocab_size
        self.review_size = self._dataset.review_size
        self.user_size = self._dataset.user_size
        self.product_size = self._dataset.product_size
        self.query_max_length = self._dataset.query_max_length
        self.vocab_distribute = self._dataset.vocab_distribute
        self.review_distribute = self._dataset.review_distribute
        self.product_distribute = self._dataset.product_distribute

        self._params = params
        self.negative_sample = self._params['negative_sample']
        self.embed_size = self._params['embed_size']
        self.window_size = self._params['window_size']
        self.max_gradient_norm = self._params['max_gradient_norm']
        self.init_learning_rate = self._params['init_learning_rate']
        self.L2_lambda = self._params['L2_lambda']
        self.net_struct = self._params['net_struct']
        self.similarity_func = self._params['similarity_func']
        self.query_weight=self._params['query_weight']
        self.global_step = tf.Variable(0, trainable=False)

        self.forward_only = forward_only

        self.print_ops = []
        if self.query_weight >= 0:
            self.Wu = tf.Variable(self.query_weight, name="user_weight", dtype=tf.float32, trainable=False)
        else:
            self.Wu = tf.sigmoid(tf.Variable(0, name="user_weight", dtype=tf.float32))

        self.context_word_idxs = []
        for i in range(2 * self.window_size):
            self.context_word_idxs.append(tf.placeholder(tf.int64, shape=[None], name="context_idx{0}".format(i)))

    def build(self):
        self._build_placeholder()
        self.loss = self._build_embedding_graph_and_loss()

        if not self.forward_only:
            self.updates = self._build_optimizer()
        else:
            self.product_scores = self.get_product_scores(self.user_idxs, self.query_word_idxs)

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_placeholder(self):
        # Feeds for inputs.
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.review_idxs = tf.placeholder(tf.int64, shape=[None], name="review_idxs")
        self.user_idxs = tf.placeholder(tf.int64, shape=[None], name="user_idxs")
        self.product_idxs = tf.placeholder(tf.int64, shape=[None], name="product_idxs")
        self.word_idxs = tf.placeholder(tf.int64, shape=[None], name="word_idxs")
        self.query_word_idxs = tf.placeholder(tf.int64, shape=[None, self.query_max_length], name="query_word_idxs")
        self.PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)

    def LSE_nce_loss(self, user_idxs, product_idxs, word_idxs, context_word_idxs):
        batch_size = array_ops.shape(word_idxs)[0]  # get batch_size
        loss = None

        # get f(s)
        word_idx_list = tf.stack([word_idxs] + context_word_idxs, 1)
        f_s, [f_W, word_vecs] = get_query_embedding(self, word_idx_list, self.word_emb, None)

        # Negative sampling
        loss, true_w, sample_w = self.LSE_single_nce_loss(f_s, product_idxs, self.product_emb,
                                                     self.product_bias, self.product_size, self.product_distribute)

        # L2 regularization
        if self.L2_lambda > 0:
            loss += self.L2_lambda * (tf.nn.l2_loss(true_w) + tf.nn.l2_loss(sample_w) +
                                      tf.nn.l2_loss(f_W) + tf.nn.l2_loss(word_vecs))

        return loss / math_ops.cast(batch_size, dtypes.float32)

    def LSE_single_nce_loss(self, example_vec, label_idxs, label_emb,
                            label_bias, label_size, label_distribution):
        batch_size = array_ops.shape(label_idxs)[0]  # get batch_size
        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(tf.cast(label_idxs, dtype=tf.int64), [batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.negative_sample,
            unique=False,
            range_max=label_size,
            distortion=0.75,
            unigrams=label_distribution))

        # get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
        true_w = tf.nn.embedding_lookup(label_emb, label_idxs)
        true_b = tf.nn.embedding_lookup(label_bias, label_idxs)

        # get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
        sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)
        sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise lables for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [self.negative_sample])
        sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec

        return self.nce_loss(true_logits, sampled_logits), true_w, sampled_w

    # return model.nce_loss(true_logits, true_logits)

    def nce_loss(self, true_logits, sampled_logits):
        "Build the graph for the NCE loss."

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_logits, labels=tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sampled_logits, labels=tf.zeros_like(sampled_logits))

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent))
        return nce_loss_tensor

    def _build_embedding_graph_and_loss(self, scope=None):
        with variable_scope.variable_scope(scope or "LSE_graph"):
            # Word embeddings.									
            init_width = 0.5 / self.embed_size
            self.word_emb = tf.Variable(tf.random_uniform(
                [self.vocab_size, self.embed_size], -init_width, init_width),
                name="word_emb")
            self.word_emb = tf.concat(axis=0, values=[self.word_emb, tf.zeros([1, self.embed_size])])
            self.word_bias = tf.Variable(tf.zeros([self.vocab_size]), name="word_b")
            self.word_bias = tf.concat(axis=0, values=[self.word_bias, tf.zeros([1])])

            # user/product embeddings.									
            self.user_emb = tf.Variable(tf.zeros([self.user_size, self.embed_size]),
                                         name="user_emb")
            self.user_bias = tf.Variable(tf.zeros([self.user_size]), name="user_b")
            self.product_emb = tf.Variable(tf.zeros([self.product_size, self.embed_size]),
                                            name="product_emb")
            self.product_bias = tf.Variable(tf.zeros([self.product_size]), name="product_b")

            # self.context_emb = tf.Variable( tf.zeros([self.vocab_size, self.embed_size]),								
            #						name="context_emb")			
            # self.context_bias = tf.Variable(tf.zeros([self.vocab_size]), name="context_b")								
            return self.LSE_nce_loss(self.user_idxs, self.product_idxs, self.word_idxs,
                                self.context_word_idxs)

    def _build_optimizer(self):
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gradients = tf.gradients(self.loss, params)

        self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                 self.max_gradient_norm)
        return opt.apply_gradients(zip(self.clipped_gradients, params),
                                         global_step=self.global_step)

    def step(self, session, input_feed, forward_only, file_writer=None, test_mode='product_scores'):
        # if not forward_only:
        #     output_feed = [self.updates,    # Update Op that does SGD.
        #                  self.loss]    # Loss for this batch.
        # else:
        #     if test_mode == 'output_embedding':
        #         output_feed = [self.user_emb, self.product_emb, self.Wu, self.word_emb, self.word_bias]
        #     else:
        #         output_feed = [self.product_scores, self.print_ops]
        #
        # outputs = session.run(output_feed, input_feed)  #options=run_options, run_metadata=run_metadata)
        #
        # if not forward_only:
        #     return outputs[1]   # loss, no outputs, Gradient norm.
        # else:
        #     if test_mode == 'output_embedding':
        #         return outputs[:4], outputs[4:]
        #     else:
        #         return outputs[0], None    # product scores to input user
        if not forward_only:
            output_feed = [self.updates,    # Update Op that does SGD.
                         self.loss]    # Loss for this batch.
        else:
            if test_mode == 'output_embedding':
                output_feed = [self.user_emb, self.product_emb, self.Wu, self.word_emb, self.word_bias]

            else:
                output_feed = [self.product_scores, self.print_ops] #negative instance output

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1]   # loss, no outputs, Gradient norm.
        else:
            if test_mode == 'output_embedding':
                return outputs[:4], outputs[4:]
            else:
                return outputs[0], None    # product scores to input user

    def get_product_scores(self, user_idxs, query_word_idx, product_idxs = None, scope = None):
        """
        Args:
            user_idxs: Tensor with shape of [batch_size] with type of int32.
            query_word_idx: Tensor with shape for [batch_size, query_max_length] with type of int32.
            product_idxs: Tensor with shape of [batch_size] with type of int32 or None.
            scope:

        Return:
            product_scores: Tensor with shape of [batch_size, batch_size] or [batch_size, len(product_vocab)]
                            with type of float32. its (i, j) entry is the score of j product retrieval by i
                            example(which is a linear combination of user and query).


        """

        with variable_scope.variable_scope(scope or "LSE_graph"):
            # get query vector
            query_vec, word_vecs = get_query_embedding(self, query_word_idx, self.word_emb, True)
            # match with product
            product_vec = None
            product_bias = None
            if product_idxs != None:
                product_vec = tf.nn.embedding_lookup(self.product_emb, product_idxs)
                product_bias = tf.nn.embedding_lookup(self.product_bias, product_idxs)
            else:
                product_vec = self.product_emb
                product_bias = self.product_bias

            print('Similarity Function : ' + self.similarity_func)

            if self.similarity_func == 'product':
                return tf.matmul(query_vec, product_vec, transpose_b=True)
            elif self.similarity_func == 'bias_product':
                return tf.matmul(query_vec, product_vec, transpose_b=True) + product_bias
            else:
                query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_vec), 1, keep_dims=True))
                product_norm = tf.sqrt(tf.reduce_sum(tf.square(product_vec), 1, keep_dims=True))
                return tf.matmul(query_vec / query_norm, product_vec / product_norm, transpose_b=True)
