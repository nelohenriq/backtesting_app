# Ollama Starter Kit

This template comes with Ollama pre-installed and ready to use with the llama3.1 model.

## One-Time Activity: Modify Path Variable

After duplicating this studio, you'll need modify the path variable so that our Studio knows where to find the Ollama installation and can execute commands. 

You can do this by editing the `.studiorc` configuration file and adding the following line to set the `PATH` variable:

```bash
export PATH="/teamspace/studios/this_studio/bin:${PATH}"
```

Once you've made this change, restart your Studio. Ollama should now be automatically added to your path, allowing you to execute Ollama commands without any issues.

## Running Ollama and Chatting with Llama3.1

To launch Ollama, open a terminal and run:

```bash
ollama serve
```

In another terminal window, you can start chatting with Llama3.1 by entering:

```bash
ollama run llama3.1
```

Simply type in your prompt to start the conversation. To end the session with Llama3.1, type `/bye`. 

## Update Ollama

Ollama will not auto-update in your lightning.ai studio. To trigger the update manually, execute the file `update_ollama.sh` by entering the following command in the terminal:

```bash
./update_ollama.sh
```
