# SimBA

This thesis introduces SimBA (Similarity Based or Similarity BAchelor Thesis),
an unsupervised Information Retrieval (IR) system that combines lexical and semantic similarity features in order to retrieve and rank relevant targets for a given input query. It was developed as a tool for verified claim retrieval, which can be regarded as a special case of an IR task based on the detection of semantically similar claims. The presented system computes embeddings of queries and targets using three different sentence encoders, averages the similarity scores of the embeddings and combines the result with a simple lexical feature based on token overlap between query and target.
SimBA showcases the expressiveness and usability of sentence embeddings in the context of semantic similarity. It demonstrates how unsupervised systems are able to perform on par with computationally expensive supervised systems for verified claim retrieval. 
Additionally, the conducted experiments indicate that differently trained sentence encoders capture different information and that it is beneficial to use them in a complementary way. Similarly, the information encoded by sentence encoders seems to differ sufficiently much from lexical information to let the system profit from the addition of a lexical feature.


## How to use

Install the required libraries and run the scripts in the script folder to produce tables and figures used in the thesis.

