import tensroflow as tf
from tensorflow.keras.layers import Layer,Embedding,Add

class EmbeddingLayer(Layer):
    def __init__(self,seqlen,embedding_dim,embedding_matrix,positional_matrix,**kwargs):
        super().__init__()
        self.seqlen=seqlen
        self.embedding_dim=self.embedding_dim
        self.embed_matrix=embedding_matrix
        self.positional_matrix=positional_matrix
        
    def build(self):
        self.position_embedding=self.add_variable(shape=(self.seqlen,self.embedding_dim),
                                                  dtype=tf.float32,
                                                  trainable=True,
                                                  initializer=self.position_matrix)
        
    def call(self,inp):
        x=Embedding(input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                    weights=[self.embedding_matrix]
                    )(inp)
        
        x=Add()[x,self.position_embedding]
        return x
        
    
