import numpy
import os
import tensorflow as tf
from tensroflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding,Add,Lambda
from attention_layer import AttentionLayer


class Transformer():
    
    def __init__(self,num_of_encoder_stack,num_of_decoder_stack,seqlen,embedding_dim,attention_dim,vocab_size,embedding_path,batch_size):
        super().__init__()
        self.num_of_encoder_stack=num_of_encoder_stack
        elf.num_of_decoder_stack=num_of_decoder_stack
        self.embedding_dim=embedding_dim
        self.seqlen=seqlen
        self.vocab_size=vocab_size
        self.embedding_path=embedding_path
        self.batch_size=batch_size
        self.attention_dim=attention_dim
        
    def fit(self,X,Y):
        tokenizer=Tokenizer(oov_token="OOV")
        tokenizer.fit_on_texts(X)
        self.tokenized_X=tokenizer.texts_to_sequences(X)
        self.tokenized_Y=tokenizer.texts_to_sequences(Y)
        self.word_index=tokenizer.word_index()
        self.embedding_matrix_generation()
        self.positional_matrix_generation()
        
    def embedding_matrix_generation(self):
        
        if self.embedding_path is not None:
            embedding_matrix=np.zeros(shape=(len(self.word_index)+1,self.embedding_dim),dtype="float32")
            embed_index={}
            with open(self.embedding_path,"r",encoding="utf-8") as file:
                for line in file:
                    vector=line.split()
                    word=vector[0]
                    values=np.asarray(vector[1:],dtype='float-32')
                    embed_index[word]=values
                    
            for word in self.word_index.keys():
                if word in embed_index.keys():
                    embedding_matrix[self.word_index[word]]=embed_index[word]
        else:
            embedding_matrix=np.random.normal(shape=(len(self.word_index)+1,self.embedding_dim),mean=0.0,stddev=1.0)
            
        self.embedding_matrix=embedding_matrix
        
    def positional_matrix_generation(self):
        position_matrix=np.zeros(shape=(self.seqlen,self.embedding_dim),dtype='float-32')
        n=10000
        
        for k in range(self.num_seq):
            for i in np.arange(int(self.embedding_dim/2)):
                denom=np.power(n,2*i/self.embedding_dim)
                position_matrix[k,2*i]=np.sin(k/denom)
                position_matrix[k,2*i+1]=np.cos(k/denom)
                
        self.position_matrix=position_matrix
        
    def build(self):
        #Building Encoder:
        inp=Input(shape=(self.seqlen),name="encoder_inp")

        x=Embedding(input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                    weights=[self.embedding_matrix],
                    name='encoder_embeddings')(inp)
        
        y=tf.Variable(shape=(self.seqlen,self.embedding_dim),
                      dtype=tf.float32,
                      name="encoder_positional_embedding",
                      trainable=True,
                      initial_value=self.position_matrix)
        
        out=Add()[x,y]
        out=AttentionLayer(seqlen=self.seqlen,hidden_dim=self.attention_dim,out_dim=self.embedding_dim,name="encoder_attention_layer")(out)
        
        
        
    