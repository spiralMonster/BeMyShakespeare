import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Add,LSTM,LayerNormalization
from attention_layer import AttentionLayer

class EncoderLayer(Layer):
    def __init__(self,encoder_seqlen,hidden_dim,out_dim,**kwargs):
        super().__init__(**kwargs)
        self.encoder_seqlen=seqlen
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.stack=None
        
    def call(self,embeddings):
        self.stack=embeddings
        out=AttentionLayer(seqlen=self.encoder_seqlen,hidden_dim=self.hidden_dim,out_dim=self.out_dim)(embeddings)
        out=Add()[out,self.stack]
        out=LayerNormalization()(out)
        self.stack=out
        out=LSTM(units=self.out_dim,kernel_initializer='he_uniform',activation='relu',return_sequences=True)(out)
        out=Add()[out,self.stack]
        out=LayerNormalization()(out)
        return out
