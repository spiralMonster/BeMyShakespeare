import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.layers import Dense

class AttentionLayer(Layer):
    def __init__(self,seqlen,hidden_dim,out_dim,**kwargs):
        super().__init__(**kwargs)
        self.seqlen=seqlen
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        
    def build(self):
        self.query=self.add_weight(
            shape=(self.seqlen,self.hidden_dim),
            dtype=tf.float32,
            name='attention_query',
            trainable=True,
            initializer='glorot_uniform'
        )
        
        self.key=self.add_weight(
            shape=(self.seqlen,self.hidden_dim),
            dtype=tf.float32,
            name='attention_key',
            trainable=True,
            initializer='glorot_uniform'
        )
        
        self.value=self.add_weight(
            shape=(self.seqlen,self.hidden_dim),
            dtype=tf.float32,
            name='attention_value',
            trainable=True,
            initializer='glorot_uniform'
        )
        
    def call(self,x):
        query_key=tf.linalg.matmul(self.query,tf.transpose(self.key))
        dense_out=Dense(self.hidden_dim,kernel_initializer='glorot_uniform',activation='sigmoid',name='attention_dense_layer')(query_key)
        value_weights=tf.linalg.matmul(dense_out,tf.transpose(self.value))
        out=tf.linalg.matmul(value_weights,x)
        return out
        
        
        
        
        