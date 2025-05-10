# CipherMind

An encrypted communication method based on LLM

## Function

- model
  - ciphermind.py - core model
  - server.py & client.py - receiver and sender, run in command line to transmit text
  - *test.py* - to test a model's repeat ability (tempporary exists)
  - collision.py - to test the collision rate (describe in cosine similarity pattern) of the base model, output its result to ./data/res/collision

- experiment
  - mmlu.py - apply the mmlu test on the given base model and tunned(lora) model, output its result to ./data/res/mmlu
  - correctness.py - to test the tranmission ability of the base model and tunned(lora) model, output its result to ./data/res/correctness
  - show.py - visualize the result of three test, namely the collision, correctness, mmlu

## Usage

> Enter corresponding directory first

- model
  - server & client

    ``` sh
    # one terminal
    python servere.py
    # another terminal
    python client.py
    ```

  - collision

    ``` sh
    python collision.py
    ```

- experiment
  - mmlu

    ``` sh
    python mmlu.py
    ```

  - correctness

    ``` sh
    python correctness.py
    ```

  - visualize

    ``` sh
    python show.py
    ```
