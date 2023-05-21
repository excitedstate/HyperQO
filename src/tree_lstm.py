import torch
import torch.nn as nn
import src.config as config


class TreeLSTM(nn.Module):
    """
        this is a tree lstm module, but binary tree only
    """

    def __init__(self, input_size: int, hidden_size: int, elementwise_affine=False):
        """
            note:
                1. Linear: y = xA^T + b
                2. LayerNorm: y = (x - mean) / (std + eps) * gamma + beta, eps in (0, 1e-5), gamma, beta in R^d
                3. Dropout: y = x * mask / (1 - p), mask ~ Bernoulli(1 - p), p in [0, 1]
            structure:
                1. fc_left, fc_right: hidden -> 5 * hidden_size
                2. fc_input: input -> 5 * hidden_size
                3. layer_norm_left, layer_norm_right, layer_norm_input: 5 * hidden_size -> 5 * hidden_size
                4. layer_norm_c: hidden_size -> hidden_size
                5. dropout: p = 0.2
        @param input_size:
        @param hidden_size:
        @param elementwise_affine:
        """
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.fc_left = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_right = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_input = nn.Linear(input_size, 5 * hidden_size)

        self.layer_norm_input = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)
        self.layer_norm_left = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)

        self.layer_norm_right = nn.LayerNorm(5 * hidden_size, elementwise_affine=elementwise_affine)

        self.layer_norm_c = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, h_left, c_left, h_right, c_right, feature):
        """
            progress:
                lstm_in = norm(fc_left(h_left)) + norm(fc_right(h_right)) + norm(fc_input(feature))
                lstm_in: 5 * hidden_size

                a, i, f1, f2, o = lstm_in.chunk(5, 1) --> tensor (hidden_size)
                c: hidden_size

            note:
                1. chunk: split the tensor into several parts, return a tuple of tensors
        @param h_left:
        @param c_left:
        @param h_right:
        @param c_right:
        @param feature:
        @return:
        """
        lstm_in = self.layer_norm_left(self.fc_left(h_left))
        lstm_in += self.layer_norm_right(self.fc_right(h_right))
        lstm_in += self.layer_norm_input(self.fc_input(feature))
        a, i, f1, f2, o = lstm_in.chunk(5, 1)

        c = a.tanh() * i.sigmoid() + \
            f1.sigmoid() * c_left + \
            f2.sigmoid() * c_right

        c = self.layer_norm_c(c)

        h = o.sigmoid() * c.tanh()
        return h, c

    def zero_h_c(self, input_dim=1):
        """
            unused
        @param input_dim:
        @return:
        """
        return torch.zeros(input_dim, self.hidden_size, device=config.DEVICE_NAME), \
            torch.zeros(input_dim, self.hidden_size, device=config.DEVICE_NAME)


class Head(nn.Module):
    def __init__(self, hidden_size):
        super(Head, self).__init__()
        # self.hidden_size = hidden_size
        self.head_layer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, 1),
                                        )
        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.head_layer(x)
        return out


class SPINN(nn.Module):
    """
        actually, this is not a typical neural network, but some modules collections
        it contains:
            1. a leaf embedding network: can transform the table to a dense vector
            2. TreeLSTM: used to get the feature of a binary tree(encoding a tree)
            3. SQL Query Encoding(Relu and sql_layer)
            4. Plan Encoding (E = EQ + Rp), i.e self.logits
    """

    def __init__(self, head_num: int, input_size: int, hidden_size: int, table_num: int, sql_size: int):
        """

        @param head_num:
        @param input_size:
        @param hidden_size:
        @param table_num:
        @param sql_size:
        """
        super(SPINN, self).__init__()

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.table_num = table_num
        self.input_size = input_size
        self.sql_size = sql_size

        self.tree_lstm = TreeLSTM(input_size=input_size, hidden_size=hidden_size)

        self.sql_layer = nn.Linear(sql_size, hidden_size)

        self.head_layer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, head_num + 1))
        self.table_embeddings = nn.Embedding(table_num, hidden_size)  # 2 * max_column_in_table * size
        # # not used, the heads will not be used
        self.heads = nn.ModuleList([Head(self.hidden_size) for _ in range(self.head_num + 1)])
        self.relu = nn.ReLU()
        self.init()

    def init(self):
        for name, param in self.named_parameters():
            if len(param.shape) == 2:
                nn.init.xavier_normal_(param)
            else:
                nn.init.uniform_(param)

    def leaf(self, alias_id):
        """
            get the embedding of the alias_id
        @param alias_id:
        @return:
        """
        table_embedding = self.table_embeddings(alias_id)
        return table_embedding, torch.zeros(table_embedding.shape, device=config.DEVICE_NAME, dtype=torch.float32)

    def input_feature(self, feature):
        """
            transform the feature into tensor
        @param feature:
        @return:
        """
        return torch.tensor(feature, device=config.DEVICE_NAME, dtype=torch.float32).reshape(-1, self.input_size)

    @staticmethod
    def sql_feature(feature):
        """
            transform the feature of sql query feature into tensor
        @param feature:
        @return:
        """
        return torch.tensor(feature, device=config.DEVICE_NAME, dtype=torch.float32).reshape(1, -1)

    def target_vec(self, target):
        """
            transform the target into tensor
        @param target:
        @return:
        """
        return torch.tensor([target] * self.head_num, device=config.DEVICE_NAME, dtype=torch.float32).reshape(1, -1)

    def tree_node(self, h_left, c_left, h_right, c_right, feature):
        """
            node output
        @param h_left:
        @param c_left:
        @param h_right:
        @param c_right:
        @param feature:
        @return:
        """
        h, c = self.tree_lstm(h_left, c_left, h_right, c_right, feature)
        return h, c

    def logits(self, encoding, sql_feature):
        """
            get the logits, as the paper said
        @param encoding: the representation of the tree root node, derived from the depth-first searching (Tree LSTM)
        @param sql_feature:
        @return:
        """
        sql_hidden = self.relu(self.sql_layer(sql_feature))
        # # Eq + R_p
        out_encoding = torch.cat([encoding, sql_hidden], dim=1)
        out = self.head_layer(out_encoding)
        return out

    def zero_hc(self, input_dim=1):
        return (torch.zeros(input_dim, self.hidden_size, device=config.DEVICE_NAME),
                torch.zeros(input_dim, self.hidden_size, device=config.DEVICE_NAME))
