# CS441_Fall2024
Class repository for CS441 on Cloud Computing taught at the University of Illinois, Chicago in Fall, 2024

The Goal of this homework is to follow up on the previous homework, and use Apache spark for distributed processing of the sliding window of tokens.

Apache Spark is a multi-language engine for executing data engineering, data science, and machine learning on single-node machines or clusters.

The main idea of embeddings is to have fixed length representations for the tokens in a text regardless of the number of tokens in the vocabulary

In Homework 1, we tokenized the string corpus and embedded them, in this homework, we would be computing the sliding window of the tokens, 
for this use case I used a rather small window size of 4 for the input token, and also a window size of 4 for the output tokens. 

I used an RNNoutputlayer, Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language.

For the logging and testing requirements, I used the scalatest, and also SLFL4J library for logging utils


## Running the Application


### Prerequisites
    -java 11
    -scala 2.12.18
    
The interface of the application is fairly simple 
It takes a single app, which should be the output of the map/reduce job.

An example file is provided in src/main/input/data.txt
#### From Jar
```java -jar <jar file> src/main/input/data.txt```

#### From sbt
```sbt run src/main/input/data.txt```

    A convienient option is to use the intellij provided options as well.

## Video Presentation

    https://youtu.be/6ZTwJu8VGJg?si=N1s6u-k3hGjrwYZ1