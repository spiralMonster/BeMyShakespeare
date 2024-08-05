import tensorflow as tf
from tensorflow.keras.layers import Layer,LSTM,Add,LayerNormalization
from attention_layer import AttentionLayer

class DecoderLayer(Layer):
    def __init__(self,decoder_seqlen,hidden_dim1,hidden_dim2,out_dim,encoder_output,**kwargs):
        super().__init__(**kwargs)
        self.decoder_seqlen=decoder_seqlen
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.out_dim=out_dim
        self.encoder_output=encoder_output
        self.stack=None
        
    def call(self,embeddings):
        self.stack=embeddings
        out=AttentionLayer(seqlen=self.decoder_seqlen,hidden_dim=self.hidden_dim1,self.out_dim=out_dim)(embeddings)
        out=Add()(out,self.stack)
        out=LayerNormalization()(out)
        self.stack=out
        out=AttentionLayer(seqlen=self.decoder_seqlen,hidden_dim=self.hidden_dim2,self.out_dim=out_dim)(out)
        out=Add()[out,self.encoder_output]
        out=Add()[out,self.stack]
        out=LayerNormalization(out)
        self.stack=out
        out=LSTM(units=self.out_dim,kernel_initializer='he_uniform',activation='relu',return_sequences=True)(out)
        out=Add()[out,self.stack]
        out=LayerNormalization()(out)
        return out
        