import importlib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import utils
import random


class Recommender:
    """
    Recommender System class
    """

    def __init__(self, config, logger, data):
        self.data = data
        self.config = config
        self.logger = logger
        self.page_size = config["page_size"]
        self.random_k = config["rec_random_k"]
        module = importlib.import_module("recommender.model")
        self.model = getattr(module, config["rec_model"])(
            config, self.data.get_user_num(), self.data.get_item_num()
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.epoch_num = config["epoch_num"]
        self.inter_df = None
        self.entropy_df = None
        self.review_df = None
        self.record_df = None
        self.inter_num = 0

        self.train_data = []
        self.record = {}
        self.positive = {}
        self.reviews = {}
        self.round_entropy = {}
        self.round_record = {}

        for user in self.data.get_full_users():
            self.record[user] = []
            self.positive[user] = []
            self.reviews[user] = []
            self.round_entropy[user] = []
            self.round_record[user] = {}

    def train(self):
        if len(self.train_data) == 0:
            return
        users = [x[0] for x in self.train_data]
        items = [x[1] for x in self.train_data]
        labels = [x[2] for x in self.train_data]

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(users), torch.tensor(items), torch.tensor(labels)
        )

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        self.model.train()

        for epoch in tqdm(range(self.epoch_num)):
            epoch_loss = 0.0
            for user, item, label in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(user, item)
                loss = self.criterion(outputs, label.float())
                # print(f"epoch:{epoch}\n loss:{loss}\n")
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.logger.info(
                f"Epoch {epoch + 1}/{self.epoch_num}, Loss: {epoch_loss / len(train_loader)}"
            )

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def swap_items(self, lst, page_size, random_k):
        total_pages = len(lst) // page_size
        lst = lst[: total_pages * page_size]
        for page in range(1, total_pages // 2 + 1):  # 只需要迭代前一半的页面
            # 计算当前页面和对称页面的开始和结束索引
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size - 1
            symmetric_start_idx = (total_pages - page) * page_size
            symmetric_end_idx = symmetric_start_idx + page_size

            # 交换random_k个item
            for k in range(1, random_k + 1):
                lst[end_idx - k], lst[symmetric_end_idx - k] = (
                    lst[symmetric_end_idx - k],
                    lst[end_idx - k],
                )

        return lst

    def add_random_items(self, user, item_ids):
        item_ids = self.swap_items(item_ids, self.page_size, self.random_k)
        return item_ids

    def get_full_sort_items_(self, user, random_=False):
        """
        Get a list of sorted items for a given user.
        """
        items = self.data.get_full_items()
        user_tensor = torch.tensor(user)
        items_tensor = torch.tensor(items)
        sorted_items = self.model.get_full_sort_items(user_tensor, items_tensor)
        if self.random_k > 0 and random_ is True:
            sorted_items = self.add_random_items(user, sorted_items)

        sorted_items_ = [item for item in sorted_items if item not in self.record[user]]
        sorted_items_ = sorted_items_ if len(sorted_items_) < 1000 else sorted_items_[:1000]
        sorted_item_names = self.data.get_item_names(sorted_items_)
        sorted_item_details = self.data.get_item_description_by_id(sorted_items_)
        return sorted_items_, sorted_item_names, sorted_item_details

    def get_item(self, idx):
        item_name = self.data.get_item_names([idx])[0]
        description = self.data.get_item_description_by_id([idx])[0]
        item = item_name + ";;" + description
        return item

    def get_inter_num(self):
        return self.inter_num

    def update_history_by_name(self, user_id, item_names):
        """
        Update the history of a given user.
        """
        item_names = [item_name.strip(" <>'\"") for item_name in item_names]
        item_ids = self.data.get_item_ids(item_names)
        self.record[user_id].extend(item_ids)

    def update_history_by_id(self, user_id, item_id):
        """
        Update the history of a given user.
        """
        self.record[user_id].append(item_id)

    def update_history_by_ids(self, user_id, item_ids):
        """
        Update the history of a given user.
        """
        self.record[user_id].extend(item_ids)

    def update_positive(self, user_id, item_names):
        """
        Update the positive history of a given user.
        """
        item_ids = self.data.get_item_ids(item_names)
        if len(item_ids) == 0:
            return
        self.positive[user_id].extend(item_ids)
        self.inter_num += len(item_ids)

    def update_positive_by_id(self, user_id, item_id, rd, type1, type2):
        """
        Update the history of a given user.
        """
        self.positive[user_id].append([item_id, rd, type1, type2])

    def update_review(self, user_id, item_id, rating, review, rd):
        self.reviews[user_id].append([rd, item_id, rating, review])

    def update_round_record(self, user_id, page, rd):
        if rd not in self.round_record[user_id]:
            self.round_record[user_id][rd] = []
            self.round_record[user_id][rd].extend(page)
        else:
            self.round_record[user_id][rd].extend(page)

    def update_round_entropy(self, user_id, rd, data):
        entropy = utils.get_entropy(self.round_record[user_id][rd], data)
        self.round_entropy[user_id].append([rd, entropy])

    def save_interaction(self):
        """
        Save the interaction history, reviews and round entropy to a csv file.
        """
        inters, review, entropy, record = [], [], [], []
        users = self.data.get_full_users()
        for user in users:
            for item in self.positive[user]:
                new_row = {"user_id": user, "item_id": item[0], "round": item[1], "type1": item[2], "type2": item[3]}
                inters.append(new_row)
            for item in self.reviews[user]:
                new_row = {"user_id": user, "item_id": item[1], "rating": item[2], "round": item[0], "review": item[3]}
                review.append(new_row)
            for item in self.round_entropy[user]:
                new_row = {"user_id": user, "round": item[0], "entropy": item[1]}
                entropy.append(new_row)
            for rd in self.round_record[user].keys():
                lt = self.round_record[user][rd]
                new_row = {"round": rd, "user_id": user, "item_ids": lt}
                record.append(new_row)

        df = pd.DataFrame(inters)
        df.to_csv(self.config["interaction_path"] + 'interaction.csv', index=False, )
        self.inter_df = df

        df_ = pd.DataFrame(review)
        df_.to_csv(self.config["interaction_path"] + 'reviews.csv', index=False, )
        self.review_df = df_

        df__ = pd.DataFrame(entropy)
        df__.to_csv(self.config["interaction_path"] + 'entropy.csv', index=False, )
        self.entropy_df = df__

        df___ = pd.DataFrame(record)
        df___.to_csv(self.config["interaction_path"] + 'record.csv', index=False, )
        self.record_df = df___

    def add_train_data(self, user, item, label):
        self.train_data.append((user, item, label))

    def clear_train_data(self):
        self.train_data = []

    def search_items(self, user, item, k=5):
        """
        Search similar items from faiss db.
        Args:
            user: str, user id
            item: str, item name
            k: int, number of similar items to return
        """
        docs = self.data.db.similarity_search(item, 100)
        search_re_items = [doc.page_content for doc in docs]
        search_re_ids = self.data.get_item_ids_exact(search_re_items)
        # search_re_ids = random.sample([id_ for id_ in search_re_ids if id_ not in self.record[user]], k)

        search_re_ids = [id_ for id_ in search_re_ids if id_ not in self.record[user]]
        search_re_ids = search_re_ids if len(search_re_ids) < k else search_re_ids[:k]

        search_re_items = [str(i) for i in self.data.get_item_names(search_re_ids)]
        search_re_details = self.data.get_item_description_by_id(search_re_ids)
        return search_re_items, search_re_ids, search_re_details
