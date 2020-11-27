import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query_projection = nn.Linear(input_dim, input_dim * hidden_dim)
        self.reduce_projection = nn.Linear(hidden_dim, 1, bias=False)
        self.eps = torch.tensor(1e-6, requires_grad=False)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.query_projection.weight)
        nn.init.uniform_(self.query_projection.bias)
        nn.init.xavier_normal_(self.reduce_projection.weight)

    def forward(self, reviews_embedding, query_embedding):
        """
        Parameters
        -----------
        reviews_embedding: shape(batch, input_dim)
        query_embedding: shape(1, input_dim) or (input_dim,)
        """
        # ------------tanh(W*q+b)------------
        projected_query = torch.tanh(self.query_projection(query_embedding))
        # shape: (1, input_dim * hidden_dim) or (input_dim * hidden_dim)
        projected_query = projected_query.view((self.input_dim, self.hidden_dim))
        # shape: (input_dim, hidden_dim)
        # ------------r*tanh(W*q+b)------------
        reviews_query_dotted_sum = reviews_embedding @ projected_query
        # shape: (batch, hidden_dim)
        # ------------(r*tanh(W_1*q+b))*W_2------------
        reviews_query_reduce_sum = self.reduce_projection(reviews_query_dotted_sum)
        # shape: (batch, 1)
        weight = torch.softmax(reviews_query_reduce_sum, dim=0)

        # # ------------exp((r*tanh(W_1*q+b))*W_2)------------
        # reviews_query_reduce_sum = torch.squeeze(torch.exp(self.reduce_projection(reviews_query_dotted_sum)), dim=1)
        # # shape: (batch,)
        # denominator = torch.sum(reviews_query_reduce_sum, dim=0)
        # denominator = torch.clamp(denominator, min=1e-4)
        # print("denominator:", denominator)
        # weight = torch.unsqueeze(reviews_query_reduce_sum / denominator, dim=1)
        # print("weight:", weight)
        # shape: (batch, 1)
        entity_embedding = torch.sum(weight * reviews_embedding, dim=0)
        # shape: (input_dim)
        return entity_embedding


class Model(nn.Module):
    def __init__(self, word_num, word_embedding_size, doc_embedding_size,
                 attention_hidden_dim):
        super(Model, self).__init__()
        self.word_num = word_num
        self.word_embedding_size = word_embedding_size
        self.doc_embedding_size = doc_embedding_size
        self.attention_hidden_dim = attention_hidden_dim

        self.word_embedding_layer = nn.Embedding(self.word_num, self.word_embedding_size, padding_idx=0)
        self.doc_embedding_layer = nn.LSTM(input_size=self.word_embedding_size,
                                           hidden_size=self.doc_embedding_size,
                                           num_layers=1,
                                           batch_first=True)
        self.attention_layer = AttentionLayer(self.doc_embedding_size, self.attention_hidden_dim)
        self.personalized_factor = nn.Parameter(torch.tensor([0.0]))
        self.gamma = nn.Parameter(torch.tensor([0.0]))

        self.local_parameters: list = [*self.attention_layer.parameters(), self.personalized_factor, self.gamma]
        self.global_parameters: list = [self.word_embedding_layer.weight, *self.doc_embedding_layer.all_weights[0]]
        # self.local_parameters: list = [self.personalized_factor, self.gamma]
        # self.global_parameters: list = [self.word_embedding_layer.weight]

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.word_embedding_layer.weight, 0.0, 0.01)
        self.doc_embedding_layer.reset_parameters()
        self.attention_layer.reset_parameters()
        nn.init.uniform_(self.personalized_factor)
        nn.init.uniform_(self.gamma)

    def set_local(self):
        for global_parameters in self.global_parameters:
            global_parameters.requires_grad = False

    def set_global(self):
        for global_parameters in self.global_parameters:
            global_parameters.requires_grad = True

    def embedding(self, words, lengths):
        word_embedding = pack_padded_sequence(self.word_embedding_layer(words), lengths,
                                              batch_first=True, enforce_sorted=False)
        _, (_, doc_embedding) = self.doc_embedding_layer(word_embedding)
        return doc_embedding.squeeze(dim=0)
        # output, (_, _) = self.doc_embedding_layer(word_embedding)
        # output, _ = pad_packed_sequence(output, batch_first=True)
        # doc_embedding = torch.max(output, dim=1).values
        # return doc_embedding

    def forward(self,
                user_reviews_words: torch.LongTensor, user_reviews_lengths: torch.LongTensor,
                item_reviews_words: torch.LongTensor, item_reviews_lengths: torch.LongTensor,
                query: torch.LongTensor,
                mode,
                negative_item_reviews_words: torch.LongTensor = None,
                negative_item_reviews_lengths: torch.LongTensor = None):
        if mode == 'output_embedding':
            item_reviews_embedding = self.embedding(item_reviews_words, item_reviews_lengths)
            query_embedding = self.embedding(query.unsqueeze(dim=0), torch.LongTensor([len(query)]))
            item_entity = self.attention_layer(item_reviews_embedding, query_embedding)
            return item_entity

        user_reviews_embedding = self.embedding(user_reviews_words, user_reviews_lengths)
        item_reviews_embedding = self.embedding(item_reviews_words, item_reviews_lengths)
        query_embedding = self.embedding(query.unsqueeze(dim=0), torch.LongTensor([len(query)]))

        user_entity = self.attention_layer(user_reviews_embedding, query_embedding)
        item_entity = self.attention_layer(item_reviews_embedding, query_embedding)
        # user_entity = torch.sum(user_reviews_embedding, dim=0)
        # item_entity = torch.sum(item_reviews_embedding, dim=0)

        query_embedding = query_embedding.squeeze(dim=0)
        personalized_model = user_entity + self.personalized_factor * query_embedding

        # positive = torch.cosine_similarity(personalized_model, item_entity, dim=0, eps=1e-10)

        if mode == 'train':
            negative_item_reviews_embedding = self.embedding(negative_item_reviews_words, negative_item_reviews_lengths)
            negative_item_entity = self.attention_layer(negative_item_reviews_embedding, query_embedding)
            # negative_item_entity = torch.sum(negative_item_reviews_embedding, dim=0)
            # negative = torch.cosine_similarity(personalized_model, negative_item_entity, dim=0, eps=1e-10)
            # pair_loss = torch.relu(self.gamma - positive + negative)
            # return torch.relu(pair_loss)
            return personalized_model.unsqueeze(dim=0), item_entity.unsqueeze(dim=0), negative_item_entity.unsqueeze(dim=0)
        elif mode == 'test':
            return personalized_model.unsqueeze(dim=0), item_entity.unsqueeze(dim=0)
