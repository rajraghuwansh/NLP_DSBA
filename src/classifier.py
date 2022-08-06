import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW

from classes import SentimentClassifier, ABSA_Dataset
from helpers import *

PATH = "bert_model.pt"


class Classifier:
    """The Classifier"""
    def __init__(self):
        self.model = SentimentClassifier(3)
        self.tokenizer = AutoTokenizer.from_pretrained("activebus/BERT_Review")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, trainfile, devfile=None):
        """
        Trains the classifier model on the training set stored in file trainfile.
        """
        train_file = pd.read_csv(trainfile, delimiter="\t",
                                 names=COLUMN_NAMES, header=None)
        valid_file = pd.read_csv(devfile, delimiter="\t",
                                 names=COLUMN_NAMES, header=None)
        train_file = generate_target_and_category(train_file)
        valid_file = generate_target_and_category(valid_file)
        train_dataset = ABSA_Dataset(data_frame=train_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        valid_dataset = ABSA_Dataset(data_frame=valid_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
        self.model = self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        self.train_network(train_dataloader, valid_dataloader, optimizer, loss_function, EPOCHS)

    def train_network(self, train_iterator, valid_iterator, optimizer, criterion, epochs):
        """
        Trains a model and validates it after each epoch. Saves the best
        performing model on the validation set.
        """
        valid_loss_min = np.Inf
        for epoch in range(1, epochs + 1):
            self.model.train()
            if epoch == 1:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

            train_loss = 0.0
            train_correct_predictions = 0
            for batch_dict in train_iterator:
                input_ids = batch_dict["input_ids"].to(self.device)
                attention_mask = batch_dict["attention_mask"].to(self.device)
                targets = batch_dict["targets"].to(self.device)
                category_dummies = batch_dict["category_dummies"].to(self.device)
                optimizer.zero_grad()
                output = self.model(input_ids, attention_mask, category_dummies)
                _, preds = torch.max(output, dim=1)

                train_correct_predictions += torch.sum(preds == targets)
                loss = criterion(output, targets)
                train_loss += loss.item()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            self.model.eval()
            valid_loss = 0.0
            valid_correct_predictions = 0
            for batch_dict in valid_iterator:
                input_ids = batch_dict["input_ids"].to(self.device)
                attention_mask = batch_dict["attention_mask"].to(self.device)
                targets = batch_dict["targets"].to(self.device)
                category_dummies = batch_dict["category_dummies"].to(self.device)
                output = self.model(input_ids, attention_mask, category_dummies)
                _, preds = torch.max(output, dim=1)

                valid_correct_predictions += torch.sum(preds == targets)
                loss = criterion(output, targets)
                valid_loss += loss.item()

            # evaluation
            train_loss = train_loss / len(train_iterator)
            valid_loss = valid_loss / len(valid_iterator)
            train_accuracy = train_correct_predictions / TRAIN_SAMPLES
            valid_accuracy = valid_correct_predictions / VALID_SAMPLES

            print(f"Epoch: {epoch}. " \
                  f"Training Loss: {train_loss:.6f}.  " \
                  f"Validation_loss: {valid_loss:.6f}. " \
                  f"Train accuracy: {train_accuracy:.2f}. " \
                  f"Valid accuracy: {valid_accuracy:.2f}.")

            # saving the model if validation loss has decreased
            if valid_loss < valid_loss_min:
                print(f"Validation loss decreased ({valid_loss_min:.6f} --> " \
                      f"{valid_loss:.6f}). Saving model..")
                torch.save(self.model.state_dict(), PATH)
                print("Model Saved")
                valid_loss_min = valid_loss

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test_file = pd.read_csv(datafile, delimiter="\t",
                                names=COLUMN_NAMES, header=None)
        test_file = generate_target_and_category(test_file)
        test_dataset = ABSA_Dataset(data_frame=test_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
        predictions = []
        predictions_dict = {0: "positive", 1: "negative", 2: "neutral"}

        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()
        for batch_dict in test_dataloader:
            input_ids = batch_dict["input_ids"].to(self.device)
            attention_mask = batch_dict["attention_mask"].to(self.device)
            category_dummies = batch_dict["category_dummies"].to(self.device)
            output = self.model(input_ids, attention_mask, category_dummies)
            _, preds = torch.max(output, dim=1)

            for prediction in preds.detach().cpu().numpy():
                predictions.append(predictions_dict[prediction])

        return predictions