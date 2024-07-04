# Fine Tuning TinyLlama for Text Generation with TRL

## Community Article
**Published June 28, 2024**  
**Author: Noa Roggendorff**

---

In this tutorial, we will walk you through the process of training a language model using the TinyLlama model and the Transformers library. We'll cover the following steps:

1. [Installing the Required Libraries](#installing-the-required-libraries)
2. [Logging into Hugging Face Hub](#logging-into-hugging-face-hub)
3. [Loading the Necessary Libraries and Models](#loading-the-necessary-libraries-and-models)
4. [Formatting the Dataset](#formatting-the-dataset)
5. [Setting Up the Training Arguments](#setting-up-the-training-arguments)
6. [Creating the Trainer](#creating-the-trainer)
7. [Training the Model](#training-the-model)
8. [Pushing the Trained Model to Hugging Face Hub](#pushing-the-trained-model-to-hugging-face-hub)

---

## 1. Installing the Required Libraries

We'll start by installing the necessary libraries using pip:

```bash
!pip install -q datasets accelerate evaluate trl
```

---

## 2. Logging into Hugging Face Hub

Next, we'll log into the Hugging Face Hub to access the required models and datasets:

```python
from huggingface_hub import notebook_login

notebook_login()
```

---

## 3. Loading the Necessary Libraries and Models

We'll import the required libraries and load the TinyLlama model and tokenizer:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

---

## 4. Formatting the Dataset

We'll define a function to format the prompts in the dataset and load the dataset:

```python
def format_prompts(examples):
    """
    Define the format for your dataset
    This function should return a dictionary with a 'text' key containing the formatted prompts.
    """
    pass

from datasets import load_dataset

dataset = load_dataset("your_dataset_name", split="train")
dataset = dataset.map(format_prompts, batched=True)

dataset['text'][2] # Check to see if the fields were formatted correctly
```

---

## 5. Setting Up the Training Arguments

We'll define a function to calculate the total number of training steps based on the number of epochs and batch size, and then set up the training arguments:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="your_output_dir",
    num_train_epochs=4,  # replace this, depending on your dataset
    per_device_train_batch_size=16,
    learning_rate=1e-4,
    save_steps=100000,  # a high number to save storage
    optim="sgd",
    optim_target_modules=["attn", "mlp"]
)
```

---

## 6. Creating the Trainer

We'll create an instance of the SFTTrainer from the trl library:

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=1024,
)
```

---

## 7. Training the Model

Finally, we'll start the training process:

```python
trainer.train()
```

---

## 8. Pushing the Trained Model to Hugging Face Hub

After the training is complete, you can push the trained model to the Hugging Face Hub using the following command:

```python
trainer.push_to_hub()
```

This will upload the model to your Hugging Face Hub account, making it available for future use or sharing.

Link : [Fine-tune TinyLlama](https://huggingface.co/blog/nroggendorff/finetune-tinyllama)
