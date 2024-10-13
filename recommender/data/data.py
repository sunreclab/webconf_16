from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from yacs.config import CfgNode

config = CfgNode(new_allowed=True)
config.merge_from_file("config.yaml")


class Data:
    """
    Data class for loading data from local files.
    """

    def __init__(self, config):
        self.config = config
        self.items = {}
        self.users = {}
        self.db = None
        self.load_items(config["item_path"])
        self.load_users(config["user_path"])
        self.load_faiss_db(config["index_name"])

    def load_faiss_db(self, index_name):
        """
        Load faiss db from local if exists, otherwise create a new one.
        """
        embeddings = OllamaEmbeddings(base_url=config['url'], model=config['emb_model_name'])
        try:
            self.db = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
            print("Load faiss db from local")
        except:
            titles = [item["title"] for item in self.items.values()]
            self.db = FAISS.from_texts(titles, embeddings, allow_dangerous_deserialization=True)
            self.db.save_local(index_name)

    def load_items(self, file_path):
        """
        Load items from local file.
        """
        cnt = 0
        import json
        with open(file_path, 'r', encoding='UTF-8') as fp:
            for line in fp:
                ln = json.loads(line.strip())
                self.items[cnt] = ln
                cnt += 1

    def load_users(self, file_path):
        """
        Load users from local file.
        """
        cnt = 0
        import json
        with open(file_path, 'r', encoding='UTF-8') as fp:
            for line in fp:
                ln = json.loads(line.strip())
                self.users[cnt] = ln
                cnt += 1
                if self.get_user_num() == self.config["agent_num"]:
                    break

    def get_full_items(self):
        return list(self.items.keys())

    def get_inter_popular_items(self):
        """
        Get the most popular items based on the number of interactions.
        """
        ids = sorted(
            self.items.keys(), key=lambda x: self.items[x]["inter_cnt"], reverse=True
        )[:3]
        return self.get_item_names(ids)

    def add_inter_cnt(self, item_names):
        item_ids = self.get_item_ids(item_names)
        for item_id in item_ids:
            self.items[item_id]["inter_cnt"] += 1

    def add_mention_cnt(self, item_names):
        item_ids = self.get_item_ids(item_names)
        for item_id in item_ids:
            self.items[item_id]["mention_cnt"] += 1

    def get_mention_popular_items(self):
        """
        Get the most popular items based on the number of mentions.
        """
        ids = sorted(
            self.items.keys(), key=lambda x: self.items[x]["mention_cnt"], reverse=True
        )[:3]
        return self.get_item_names(ids)

    def get_item_names(self, item_ids):
        names = []
        for item_id in item_ids:
            try:
                names.append("<" + self.items[item_id]["title"] + ">")
            except:
                names.append("<None>")
        return names

    def get_item_ids(self, item_names):
        item_ids = []
        for item in item_names:
            for item_id, item_info in self.items.items():
                if item_info["title"] in item:
                    item_ids.append(item_id)
                    break
        return item_ids

    def get_item_ids_exact(self, item_names):
        item_ids = []
        for item in item_names:
            for item_id, item_info in self.items.items():
                if item_info["title"] == item:
                    item_ids.append(item_id)
                    # break
        return item_ids

    def get_full_users(self):
        return list(self.users.keys())

    def get_user_names(self, user_ids):
        return [self.users[user_id]["username"] for user_id in user_ids]

    def get_user_ids(self, user_names):
        user_ids = []
        for user in user_names:
            for user_id, user_info in self.users.items():
                if user_info["username"] == user:
                    user_ids.append(user_id)
                    break
        return user_ids

    def get_user_num(self):
        """
        Return the number of users.
        """
        return len(self.users.keys())

    def get_item_num(self):
        """
        Return the number of items.
        """
        return len(self.items.keys())

    def get_item_description_by_id(self, item_ids):
        """
        Get description of items by item id.
        """
        descriptions = []
        from string import Template
        desc_temp = ('This product is available in the Amazon online store ${store}.'
                     + 'The Name of the product is ${title}.'
                     + 'It falls under the Main category (i.e., domain) of ${main_category}.'
                     + 'The price of this product is ${price}.'
                     + 'The product has received ${rating_number} ratings and has an average rating (i.e., Rating of the product shown on the product page) of ${average_rating}.'
                     + 'Here are the Bullet-point format features of the product: ${features}'
                     + 'Here are the description of the product: ${description}'
                     + 'Here are the other details: ${details}'
                     )
        for item_id in item_ids:
            descriptions.append(
                Template(desc_temp).substitute(
                    {'store': self.items[item_id]["store"],
                     'title': self.items[item_id]["title"],
                     'main_category': self.items[item_id]["main_category"],
                     'price': self.items[item_id]["price"],
                     'rating_number': self.items[item_id]["rating_number"],
                     'average_rating': self.items[item_id]["average_rating"],
                     'features': self.items[item_id]["features"],
                     'description': self.items[item_id]["description"],
                     'details': self.items[item_id]["details"]
                     }
                )
            )

        return descriptions

    def get_item_description_by_name(self, item_names):
        """
        Get description of items by item name.
        """
        item_descriptions = []
        for item in item_names:
            found = False
            for item_id, item_info in self.items.items():
                if item_info["title"] == item.strip(" <>"):
                    item_descriptions.append(item_info["description"])
                    found = True
                    break
            if not found:
                item_descriptions.append("")
        return item_descriptions

    def get_category_by_id(self, item_ids):
        """
        Get category of items by item id.
        """
        return [self.items[item_id]["main_category"] for item_id in item_ids]