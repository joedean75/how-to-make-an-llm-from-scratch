# %%
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction")
        return
    
    with urllib.request.urlopen(url) as response:           #1 Downloads the file
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:     #2 Unzips the file
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)           #3 Adds a .tsv file extension
    print(f"File downloaded and saved as {data_file_path}")



download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
# %%
import pandas as pd
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]

)
df  #1 Renders the data frame in a Jupyter notebook. Alternatively, use print(df).
# %%
print(df["Label"].value_counts())
# %%
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]   #1 Counts the instances of “spam”
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )                                               #2 Randomly samples “ham” instances to match the number of “spam” instances
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])                                              #3 Combines ham subset with “spam”
    return balanced_df
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
# %%
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam":1})

# %%
def random_split(df, train_frac, validation_frac):
    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)            #1 Shuffles the entire DataFrame
    train_end = int(len(df) * train_frac)   #2 Calculates split indices
    validation_end = train_end + int(len(df) * validation_frac)


    #3 Splits the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


train_df, validation_df, test_df = random_split(
    balanced_df, 0.7, 0.1
)           #4 Test size is implied to be 0.2 as the remainder.
# %% padding example
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
# %%
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# %% setting up pytorch spam data set class
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        #1  Pretokenizes texts
        
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            #2  Truncates sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
            #3 Pads sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * 
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
# %% creating batches for trainging set
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

# %% seeing max length of train dataset
print(train_dataset.max_length)
# %% creating batches for validation and trainging sets
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
# %%
