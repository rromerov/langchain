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
