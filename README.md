# An AI chatbot
## A simple implementation of a contextual Chatbot with pytorch

- The chatbot approach is based on this article and implemented using pytorch:
[https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077)

- Implementing this chat bot is quite easy and it provides beginners with a basic understanding of chatbots.

- The chatbot is based on the following architecture:
    - A Feed forward neural network with two hidden layers
- To customize the chatbot, for your own use, just modify the 'intents.json' file with possible patterns and responses and re-run the training process.

### Installation
#### Create a virtual environment:
I personally prefer virtual env:
```console
mkdir chatbot
cd chatbot
mkvirtualenv chatbot
```
Note: the virtual environment is created in the current directory and becomes immediately activated.

#### Install dependencies:
```console
pip install -r requirements.txt
```

Most times you'll tend to get an error for first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

### Usage
Run 
```console
python train.py
```
This will dump the trained model to 'data.pth'. And then run
```console
python chat.py
```

#### Customize the responses
Have a look at [intents.json](intents.json). You can customize it according to what you want. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever you modify the intents.json file.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```
