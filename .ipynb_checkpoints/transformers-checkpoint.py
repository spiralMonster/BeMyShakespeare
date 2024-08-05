import numpy
import os
import tensorflow as tf
from tensroflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input,Concatenate,Dense
from transformers_embedding_layer import EmbeddingLayer
from attention_layer import AttentionLayer
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer
from keras.models import Model



class Transformer():
    
    def __init__(self,num_of_encoder_decoder_stack,encoder_seqlen,decoder_seqlen,encoder_hidden_dim,
                 decoder_hidden_dim,embedding_dim,vocab_size,embedding_path,dense_units,batch_size):
        super().__init__()
        self.num_of_encoder_decoder_stack=num_of_encoder_decoder_stack
        self.embedding_dim=embedding_dim
        self.encoder_seqlen=encoder_seqlen
        self.decoder_seqlen=decoder_seqlen
        self.encoder_hidden_dim=encoder_hidden_dim #list
        self.decoder_hidden_dim=decoder_hidden_dim #list
        self.vocab_size=vocab_size
        self.embedding_path=embedding_path
        self.dense_units=self.dense_units #List-> only upto last second layer
        self.batch_size=batch_size
        
        
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
        position_matrix=np.zeros(shape=(self.embed_seqlen,self.embedding_dim),dtype='float-32')
        n=10000
        
        for k in range(self.embed_seqlen):
            for i in np.arange(int(self.embedding_dim/2)):
                denom=np.power(n,2*i/self.embedding_dim)
                position_matrix[k,2*i]=np.sin(k/denom)
                position_matrix[k,2*i+1]=np.cos(k/denom)
                
        self.position_matrix=position_matrix
        
    def build(self):
        inp_encoder=Input(shape=(self.encoder_seqlen),dtype=tf.float32,name='encoder_input')
        inp_decoder=Input(shape=(self.seqlen),dtype=tf.float32,name='decoder_input')
        
        #embedding matrices:
        embed_matrix=self.embedding_matrix
        position_embed_matrix=self.position_matrix
        
        #embeddings:
        embed_encoder=EmbeddingLayer(seqlen=self.encoder_seqlen,
                                     embedding_dim=self.embedding_dim,
                                     embedding_matrix=embed_matrix,
                                     positional_matrix=position_embed_matrix,
                                     name='enocder_embedding')(inp_encoder)
        
        embed_decoder=EmbeddingLayer(seqlen=self.decoder_seqlen,
                                     embedding_dim=self.embedding_dim,
                                     embedding_matrix=embed_matrix,
                                     positional_matrix=position_embed_matrix,
                                     name='decoder_embedding')(inp_decoder)
        
        out_decoder=Concatenate(axis=-1)[embed_decoder,tf.zeros(shape=(self.encoder_seqlen-self.decoder_seqlen,self.embedding_dim),dtype=tf.float32)]
        out_encoder=embed_encoder
        self.decoder_seqlen=self.encoder_seqlen
        
        for ind in range(self.num_of_encoder_decoder_stack):
            
            out_encoder=EncoderLayer(encoder_seqlen=self.encoder_seqlen,
                                     hidden_dim=self.encoder_hidden_dim[0],
                                     out_dim=self.embedding_dim,
                                     name=f'encoder_stack_{ind+1}')(out_encoder)
            
            out_decoder=DecoderLayer(decoder_seqlen=self.decoder_seqlen,
                                     hidden_dim1=self.encoder_hidden_dim[0],
                                     hidden_dim2=self.encoder_hidden_dim[1],
                                     out_dim=self.embedding_dim,
                                     encoder_output=out_encoder,
                                     name=f'decoder_stack_{ind+1}')(out_decoder)
            
        for ind in range(len(self.dense_units)):
            out_decoder=Dense(units=self.dense_units[ind],kernel_initializer='he_uniform',activation='relu')(out_decoder)
            
        
        out_decoder=Dense(units=self.vocab_size,kernel_initializer='glorot_uniform',activation='softmax')(out_decoder)
        
        Transformer=Model(inputs=[inp_encoder,inp_decoder],outputs=[out_decoder])
        
        return Transformer
        
        
        
    
            
            
            
        



        
       
        




        
        
        
        
        
        
    