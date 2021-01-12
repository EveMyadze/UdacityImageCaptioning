import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        #inherits from the DecoderRNN class
        super(DecoderRNN, self).__init__()
        
        #embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        #linear layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    #accepts as input Pytorch tensor features and PyTorch tensor corrsponding to last batch of captions
    def forward(self, features, captions):
        
        #last batch of captions
        captions = captions[:, :-1]
        
        #outputs tensor of word embeddings 
        embeddings = self.word_embeddings(captions)
        
      
        #concatenates vector of image features and word embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
       
        #pass inputs through lstm
        out, hidden = self.lstm(inputs)
        
        #pass outputs of lstm through linear layer
        outputs = self.fc(out)
        
        return outputs
 

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
       
        #initialize empty list for predicted sentence
        output = []
        
       
        for _ in range(max_len):
            #pass input through lstm      
            out, states = self.lstm(inputs, states)
            
            #pass output of lstm through linear layer
            outputs = self.fc(out)
                  
            #eliminates any dimension with size 1
            outputs = outputs.squeeze(1)
            
            #get index of max item in distribution of scores
            predict_idx = outputs.max(1)[1]
            
            #adds to empty list the word embedding associated with that index
            output.append(predict_idx.item())
            
            #creates new input for next iteration
            inputs = self.word_embeddings(predict_idx)
            inputs = inputs.unsqueeze(1)
        
        return output