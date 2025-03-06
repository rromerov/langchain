This repository contains information about LangChain.

## What is LangChain?
LangChain is an open source orchestration framework for the development of applications using large language models (LLMs). Available in both Python- and Javascript-based libraries, LangChain’s tools and APIs simplify the process of building LLM-driven applications like chatbots and virtual agents. More information can be read here: [LangChain](https://www.ibm.com/think/topics/langchain)


## LLMs
Are pretained models that can be used to generate text. They are trained on large datasets and can generate text that is coherent and contextually relevant to the user initial input or most commonly known as **prompt**.

If we for example give a prompt like:

```
Q: What is the capital of France is _____
```
The model will generate a response like:
```
A: Paris.
```
Under the hood, the LLM estiamtes the probability of a sequence of word(s) given a previous sequence of words.

> **Note:** The model makes predictions based on **tokens**, not words. A token represents an atomic unit of text, which can be a word, a subword, or a character.

The tokenization process is done by a tokenizer, which is a component that converts text into tokens.

LLMs use *transformer neural network arquitecture* to generate text. The transformer model is a deep learning model that is designed to handle sequential data using attention mechanisms. The transformer model is composed of an encoder and a decoder. The encoder processes the input sequence and generates a representation of it, which is then used by the decoder to generate the output sequence. Transformers are designed to understand the context off each word in a sentence by considering it in relation to every other word.

To train these models you need a large dataset, a lot of computational power and time. That's why it's easier to use a pre-trained model. LLMs aren't the same as Supervised ML models, they need to assemble input and output based on the training data. Which could be strategies such as removing the last word of a sentence, thus having an initial input and a target output.

> **Note:** RLHF (Reinforcement Learning from Human Feedback) is a technique that can be used to fine-tune a pre-trained model using human feedback. This technique allows the model to learn from human feedback in the form of rewards, which can be used to improve the model's performance.

## For chat format the following roles are defined:
| Role     | Description      |
| ------------- | ------------- |
| System | Used for the instructions and framing of the task |
| User | Actual task of question |
| Assistant | Output of the model |

## Prompt engineering techniques
> **Note**: Using prompt engineering techniques work better if a combination of some them is used.

### Zero-shot Prompting
This technique consists of simply instructing the model in order to do the desired request.

### Chain-of-thought
Give the model the time to think before producing the output. This is done by giving the model a sequence of tokens (an instruction) that are not part of the prompt, but that are used to give the model time to think before producing the output.

Example:
```
Think step by step.
Who could you consider to be the most famous person in the world?
```
Here the first sentence is the instruction and the second sentence is the prompt.

### Retrieval-Augmented Generation
This technique consists of finding relevant pieces of information, also known as context, from different sources and using them to generate the output. This technique is useful when the model generates responses that are not correct.

### Tool Calling
Consist of prepending the prompt with external functions that the model can use to generate the output. This should include a description of the function and how to signal in the output. Particularly useful for examples such as math problems.

Example:
```
Tools:

- calculator: This tool accepts math expressions and returns their result.

- search: This tool accepts search engine queries and returns the first search 
result.

If you want to use tools to arrive at the answer, output the list of tools and
inputs in CSV format, with this header row `tool,input`.
Who could you consider to be the most famous person in the world?
```

### Few-Shot Prompting
This focus is based on providing some questions and answers, which helps the model to learn to perform a new task withoyt the need of additional training or fine-tunning.

This approach counts with 2 options:

- Static few-shot prompting: The model is trained with a fixed set of questions and answers.
- Dynamic few-shot prompting: The model is trained with a dynamic set of questions and answers by picking the most relevant exmaples for each new query.

## Why LangChain is important?
LangChain is important because it provides a set of tools and APIs that simplify the process of building LLM-driven applications. These tools and APIs allow developers to focus on the application logic and user experience, rather than the complexities of working with LLMs.

By using LangChain, developers can quickly build and deploy LLM-driven applications, such as chatbots and virtual agents, without having to worry about the underlying infrastructure and implementation details. **LangChain** allows to use prompt techniques to improve the model's performance and generate more accurate and relevant responses.

Major LLM providers have their own syntax and APIs, which can be difficult to learn and use. LangChain provides a unified interface that abstracts away these differences, making it easier for developers to work with different LLMs.

> Note: Exercises will be done using Llama3 locally, firtly we will need to have installed ollama.

Then we will have to use CLI and run the following command:
```powershell
ollama run llama3
```

This will download Llama3 in the current directory and run it.

From there we can use :
```powershell
ollama pull llama3
```
> **Note**: llama3 latest does not support tools. As mentioned [here](https://ollama.com/search?c=tools), you could try instead with llama3.1

The next part of this notes will be located in the file `getting_started.ipynb`, as will go mainly with the exercises and examples of how to use Llama3. Click [here](getting_started.ipynb) to go to the file.

## RAGs
Simple models can be useful for certain tasks however, under most scenarios using a retrieval-augmented generation (RAG) model can be more beneficial. This model is a combination of a retriever and a generator, where the retriever finds relevant information from a knowledge source and the generator uses this information to generate the output. This technique is useful when the model generates responses that are not correct or there's information that the model is not aware of (trained).

This could include:
- Private data. As this information isn't publicly available, the model can't be trained on it.
- Current events. As the model is trained on historical data, it may not be aware of current events. As training data is a costly and time-consuming process, thus it may be trained until a certain date, this results in what is called the *knowledge cutoff*.

What it will end up happening is that the model most likely will hallucinate information, which means that it will generate information that is not correct or relevant to the prompt. Adapting the propmt won't resolve the issue either, because it relies on the model's current knowledge.

The main issue with making this data available to these models is a quantity problem. Thanks to RAGs models, you can pick the right information from a knowledge source and use it to generate the output.

To overcome this challenge it involves 2 steps:
1. **Indexing** your documents, basically preprocessing them in such a way that the application can find the most relevant information for each question.
2. **Retrieving** this external data and using it as context for the LLM to generate an accurate output based on your data.

If you for example would like to answer questions based on a document you can follow these steps:
1. *Extract* all the text from this particular document.
2. *Split* it into manageable chunks.
3. *Convert** these chunks into numbers that computers can understand.
4. *Store* these number representations of your text somewhere that makes it easy and fast to retrieve the relevant sections of your document to answer a given question.

Can be seen as the following table:
| Document | Convert to text | Split into chunks | Convert to numbers and store |
| ------------- | ------------- | ------------- | ------------- |
| Example.PDF | Text | Chunk 1 <br> Chunk 2 <br> Chunk 3 | Vector store <br> [0.1,0.2,0.5] |

This table exemplifies the preprocessing and transformation of the documents,this process is alsog known as **ingestion**. Put it in simple terms, it's converting your documents into numbers that computers can understand and analyze, and store them in a special type of database for efficient retrieval. These numbers are actually called **embeddings**, and the special type of database is known as a **vector store**.

As you may already know, machines don't work with text, they work only with numbers, **embedding** refers to representing text as a (long) sequence of numbers, also known as a vector. The benefit of vectors is actually that you can use them to do math operations.

Let's see an example of an embedding by using 3 sentences:
1. "What a sunny day."
2. "Such bright skies today."
3. "I haven’t seen a sunny day in weeks."

From there we count the number of unique words in the sentences, which are:
- What
- a
- sunny
- day
- such
- bright
- skies
- today
- I
- haven't
- seen
- in
- weeks

For each sentence, go word by word and asssign the number 0 if it's not present, 1 if used once in the sentencem 2 if present twicem and so on.

Then we can represent each word as a number, for example:
| Word | What a sunny day | Such bright skies today | I haven’t seen a sunny day in weeks |
| ------------- | ------------- | ------------- | ------------- |
| What | 1 | 0 | 0 |
| a | 1 | 0 | 1 |
| sunny | 1 | 0 | 1 |
| day | 1 | 0 | 1 |
| such | 0 | 1 | 0 |
| bright | 0 | 1 | 0 |
| skies | 0 | 1 | 0 |
| today | 0 | 1 | 0 |
| I | 0 | 0 | 1 |
| haven't | 0 | 0 | 1 |
| seen | 0 | 0 | 1 |
| in | 0 | 0 | 1 |
| weeks | 0 | 0 | 1 |

In this example the embedding for *I haven't seen a sunny day in weeks* would be:
```
[0,0,1,1,1,1,0,0,1,1,1,1,1]
```
This is called a **bag-of-words** model, and these embeddings are called **sparse embeddings** or sparse vectors, because a lot of the numbers in this vector are zeros.

This model can be used for:

- **Keyword search**: You can find which documents contain a given word or words.

- **Classification of documents**: You can calculate embeddings for a collection of examples previously labeled as email spam or not spam, average them out, and obtain average word frequencies for each of the classes (spam or not spam). Then, each new document is compared to those averages and classified accordingly.

> **Note**: This model doesn't have awareness of the meaning, only the actual words used. If we check the embeddings fro *sunny day and bright skies* look very different. In fact they have no words in common, even though we know they have a similar meaning. Or for the email classification problem, a would-be spammer can trick the filter by replacing common "spam words" with their synonyms. **Semantic embeddings** address this limitation by using numbers to represent the meaining of the text, insted of the exact words found in the text.

## LLM-based embeddings
Embedding model can be seen as an offshot from the training process of LLMs. As discussed earlier LLMs training process enables them to complete a prompt (input) qith the most appropiate continuation (output). This capability stems from an understanding of the meaining of words and sentences in the context surrounding text, learned from how words are used together in the training texts. This **understanding** of the meaning (or semantics) of the prompt can be extracted as a numeric representation (or embedding) of the input text.

An **embedding model** then is an algorithm that takes a piece of text and putputs a numeric representation of the meaning of that text. Respresented as a long list of floating-point (decimal numbers), usually between 100 and 2000 numbers, or dimensions. This is called a **dense embedding** or **dense vector**. That is the opposite of a sparse vector, because most of the numbers in this vector are not zeros.

 > **Note**: Never compare embeddings generated by different models, as they are not compatible. The embeddings are unique to the model that generated them even if they share the same size.

 ### Semantic embeddings explained
 If we for example have three words: *lion*, *pet*, and *dog*. If we thinks which words have similar characteristics, the most obvious answer will be *pet* and *dog*. As humans we understand this, however machines don't so in order to translate them into the language of computers, which is numbers.

 Let's see this example:

 ![Semantic representation of words](semanticRepWords.png)
