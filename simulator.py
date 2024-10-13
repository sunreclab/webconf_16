import random
import time
import json
import math
import dill
import faiss
import numpy as np
from typing import List
from datetime import datetime
import logging
import argparse
import pandas as pd
from yacs.config import CfgNode
from tqdm import tqdm
import concurrent.futures
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from recommender.recommender import Recommender
from recommender.data.data import Data
from agents import *
from utils import utils, message
from utils.message import Message
from agents.userMemory import RecAgentMemory, RecAgentRetriever
import shutup
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.basicConfig(level=logging.ERROR, encoding='utf-8')
shutup.please()


class Simulator:
    """
    Simulator class for running the simulation.
    """

    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt = 0
        self.round_msg: List[Message] = []
        self.active_agents: List[int] = []  # active agents in current round
        self.active_agent_threshold = config["active_agent_threshold"]
        self.active_method = config["active_method"]
        self.file_name_path: List[str] = []
        self.now = datetime.now()
        self.rec_stat = message.RecommenderStat(
            tot_user_num=0,
            cur_user_num=0,
            tot_item_num=0,
            inter_num=0,
            rec_model=config["rec_model"],
            pop_items=[],
        )

    def get_file_name_path(self):
        return self.file_name_path

    def load_simulator(self):
        """Load and initiate the simulator."""
        self.round_cnt = 0
        self.data = Data(self.config)
        self.agents = self.agent_creation()
        self.recsys = Recommender(self.config, self.logger, self.data)
        self.logger.info("Simulator loaded.")

    def save(self, save_dir_name):
        """Save the simulator status of current round"""
        utils.ensure_dir(save_dir_name)
        ID = utils.generate_id(self.config["simulator_dir"])
        file_name = f"{ID}-Round[{self.round_cnt}]-AgentNum[{self.config['agent_num']}]-{self.now.strftime('%Y-%m-%d-%H_%M_%S')}"
        self.file_name_path.append(file_name)
        save_file_name = os.path.join(save_dir_name, file_name + ".pkl")

        with open(save_file_name, "wb") as f:
            dill.dump(self.__dict__, f)

        self.logger.info("Current simulator Save in: \n" + str(save_file_name) + "\n")
        self.logger.info("Simulator File Path (root -> node): \n" + str(self.file_name_path) + "\n")
        cpkt_path = os.path.join(self.config["ckpt_path"], file_name + ".pth")
        self.recsys.save_model(cpkt_path)
        self.logger.info("Current Recommender Model Save in: \n" + str(cpkt_path) + "\n")

        """Save failed_num"""
        failed_num = []
        for aid in self.agents:
            new_row = {"user_id": self.agents[aid].id, "failed_num": self.agents[aid].failed_num}
            failed_num.append(new_row)
        df = pd.DataFrame(failed_num)
        df.to_csv(self.config["interaction_path"] + 'failed_num.csv', index=False, )

    @classmethod
    def restore(cls, restore_file_name, config, logger):
        """Restore the simulator status from the specific file"""
        with open(restore_file_name + ".pkl", "rb") as f:
            obj = cls.__new__(cls)
            obj.__dict__ = dill.load(f)
            obj.config, obj.logger = config, logger
            return obj

    def relevance_score_fn(self, score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # This will differ depending on a few things:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - score / math.sqrt(2)

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        # # Define your embedding model
        embeddings_model = OllamaEmbeddings(base_url=self.config['url'], model=self.config['emb_model_name'])
        # # Initialize the vectorstore as empty
        embedding_size = self.config['emb_size']
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {},
                            relevance_score_fn=self.relevance_score_fn, normalize_L2=True, )

        return RecAgentRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=5)

    def check_active(self, index: int):
        # If agent's previous action is completed, reset the event
        agent = self.agents[index]
        if (
                self.active_agent_threshold
                and len(self.active_agents) >= self.active_agent_threshold
        ):
            return False

        active_prob = agent.get_active_prob(self.active_method)
        if np.random.random() > active_prob:
            agent.no_action_round += 1
            return False
        self.active_agents.append(index)
        agent.no_action_round = 0
        return True

    def global_message(self, message_: str):
        for i, agent in self.agents.items():
            agent.memory.add_memory(message_, self.now)

    def one_step(self, agent_id):
        """Run one step of an agent."""
        if not self.check_active(agent_id):
            return [Message(agent_id=agent_id, action="NO_ACTION", content="No action.")]

        agent = self.agents[agent_id]
        name = agent.name
        message = []
        page = -1
        rec_item_ids, rec_items, rec_item_details = self.recsys.get_full_sort_items_(agent_id)
        leave = False

        while not leave:
            choice, observation = agent.take_action(self.now)
            leave_ = False
            while not leave_:
                if "RECOMMENDER" in choice:
                    self.recsys.update_positive_by_id(agent_id, 'in', rd=self.round_cnt, type1=0, type2=0)

                    if (page + 1) * self.recsys.page_size < len(rec_items):
                        page = page + 1
                    else:
                        self.logger.info("No more items.")
                        self.logger.info(f"{name} leaves the recommender system.")
                        message.append(Message(agent_id=agent_id, action="RECOMMENDER",
                                               content=f"No more items. {name} leaves the recommender system.", ))
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="RECOMMENDER",
                                    content=f"No more items. {name} leaves the recommender system.", ))
                        leave_ = True
                        leave = True
                        continue

                    this_page = rec_items[page * self.recsys.page_size: (page + 1) * self.recsys.page_size]
                    this_page_ids = rec_item_ids[page * self.recsys.page_size: (page + 1) * self.recsys.page_size]
                    this_page_details = rec_item_details[page * self.recsys.page_size: (page + 1) * self.recsys.page_size]

                    self.logger.info(f"{name} recommended {this_page}.")
                    message.append(
                        Message(agent_id=agent_id, action="RECOMMENDER", content=f"{name} recommended {this_page}.", )
                    )
                    self.round_msg.append(
                        Message(agent_id=agent_id, action="RECOMMENDER", content=f"{name} recommended {this_page}.", )
                    )

                    # 记录本轮推荐的items
                    self.recsys.update_round_record(agent_id, page=this_page_ids, rd=self.round_cnt)
                    # 下一次生成推荐列表之前排除
                    self.recsys.update_history_by_ids(agent_id, this_page_ids)

                    observation = (f"{name} is browsing the Amazon recommender system." +
                                   f" {name} is recommended {this_page}.")
                    choice, action = agent.take_recommender_action(observation, self.now)

                    if "VIEW" in choice:
                        # safe model.
                        if not isinstance(action, int):
                            action = 1
                        position = action - 1

                        item_name = this_page[position]
                        item_id = this_page_ids[position]
                        item_detail = this_page_details[position]

                        self.logger.info(f"{name} views {item_name}, in the Amazon recommendation page.")
                        message.append(Message(agent_id=agent_id, action="VIEW",
                                               content=f"{name} views {item_name}, in the Amazon recommendation page.", ))
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="VIEW",
                                    content=f"{name} views {item_name}, in the Amazon recommendation page.", ))

                        agent.memory.save_context({}, {
                            agent.memory.add_memory_key: f"{name} viewed {item_name}, in the Amazon recommendation page.", }, )

                        agent.update_viewed_history(item_name)
                        # 记录交互域type1和交互类型type2
                        self.recsys.update_positive_by_id(agent_id, item_id, rd=self.round_cnt, type1=0, type2=1)
                        # # 记录点击项目，用于在下一次生成推荐列表之前排除
                        # self.recsys.update_history_by_id(agent_id, item_id)

                        observation = (f"{name} is viewing the selected product." + item_detail)
                        choice, action = agent.take_rec_view_action(observation, self.now)

                        # 用于推荐模型训练的正负(hard)反馈数据
                        for i in range(self.recsys.page_size):
                            if i == position:
                                if "BUY" in choice or "CART" in choice:
                                    self.recsys.add_train_data(agent_id, this_page_ids[i], 1)
                                else:
                                    self.recsys.add_train_data(agent_id, this_page_ids[i], 0)

                        if "BUY" in choice:
                            self.logger.info(f"{name} buys {item_name}")
                            message.append(
                                Message(agent_id=agent_id, action="BUY", content=f"{name} buys {item_name}.", )
                            )
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="BUY", content=f"{name} buys {item_name}.", )
                            )
                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{name} bought {item_name}.", }, )

                            self.recsys.update_positive_by_id(agent_id, item_id, rd=self.round_cnt, type1=0, type2=3)
                            agent.update_purchased_history(item_name)

                            # getting review
                            observation = (f"{name} have bought the recommended product {item_name}." + item_detail)
                            rating, review = agent.generate_bought_review(observation, self.now, item_detail)
                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{name} gave an rating score {rating} of {item_name}. And reviewed: {review}", }, )

                            self.recsys.update_review(agent_id, item_id, rating, review, rd=self.round_cnt)
                            leave_ = True
                            leave = True
                            continue

                        elif "CART" in choice:
                            self.logger.info(f"{name} adds {item_name} to the shopping cart.")
                            message.append(Message(agent_id=agent_id, action="CART",
                                                   content=f"{name} adds {item_name} to the shopping cart.", ))
                            self.round_msg.append(Message(agent_id=agent_id, action="CART",
                                                          content=f"{name} adds {item_name} to the shopping cart.", ))
                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{name} adds {item_name} to the shopping cart.",
                                agent.memory.now_key: self.now, }, )

                            agent.update_cart({"itme_id": item_id, "item_name": item_name}, "add")
                            self.recsys.update_positive_by_id(agent_id, item_id, rd=self.round_cnt, type1=0, type2=2)
                            leave_ = True
                            continue

                        elif "SEARCH" in choice:
                            self.logger.info(f"{name} leaves recommender system and goes to the search engine.")
                            message.append(
                                Message(agent_id=agent_id, action="VIEW GT SEARCH",
                                        content=f"{name} leaves recommender system and goes to the search engine."))
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="VIEW GT SEARCH",
                                        content=f"{name} leaves recommender system and goes to the search engine."))
                            choice = "SEARCH"
                            continue

                        elif "LEAVE" in choice:
                            self.logger.info(f"{name} leaves Amazon.")
                            message.append(Message(agent_id=agent_id, action="VIEW", content=f"{name} leaves Amazon."))
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="VIEW", content=f"{name} leaves Amazon."))
                            leave_ = True
                            leave = True
                            continue

                        else:
                            leave_ = True
                            continue

                    elif "NEXT" in choice:
                        self.logger.info(f"{name} looks next page.")
                        # for i in range(self.recsys.page_size):
                        #     self.recsys.add_train_data(
                        #         agent_id, item_ids[page * self.recsys.page_size + i], 0
                        #     )
                        message.append(
                            Message(agent_id=agent_id, action="RECOMMENDER", content=f"{name} looks next page.", )
                        )
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="RECOMMENDER", content=f"{name} looks next page.", )
                        )
                        choice = 'RECOMMENDER'
                        continue

                    elif "SEARCH" in choice:
                        self.logger.info(f"{name} leaves recommender system and go to the search engine.")
                        message.append(
                            Message(agent_id=agent_id, action="RECOMMENDER GT SEARCH",
                                    content=f"{name} leaves recommender system and go to the search engine.")
                        )
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="RECOMMENDER GT SEARCH",
                                    content=f"{name} leaves recommender system and go to the search engine.")
                        )
                        choice = "SEARCH"
                        continue

                    elif "LEAVE" in choice:
                        self.logger.info(f"{name} leaves Amazon.")
                        message.append(Message(agent_id=agent_id, action="RECOMMENDER",
                                               content=f"{name} leaves Amazon.", ))
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="RECOMMENDER", content=f"{name} leaves Amazon.")
                        )
                        leave_ = True
                        leave = True

                    else:
                        leave_ = True
                        continue

                elif "SEARCH" in choice:
                    self.recsys.update_positive_by_id(agent_id, 'in', rd=self.round_cnt, type1=1, type2=0)

                    observation_ = f"{name} comes in Amazon search engine."
                    item = agent.search_item(observation_, self.now)
                    if len(item) == 0:
                        leave_ = True
                        continue

                    search_re_items, search_re_ids, search_re_details = self.recsys.search_items(agent_id, item, k=5)
                    self.logger.info(f"{name} searches for {item} in Amazon search engine, and get {search_re_items}.")
                    message.append(
                        Message(agent_id=agent_id, action="SEARCH",
                                content=f"{name} searches for {item} in Amazon search engine, and get {search_re_items}.", )
                    )
                    self.round_msg.append(
                        Message(agent_id=agent_id, action="SEARCH",
                                content=f"{name} searches for {item} in Amazon search engine, and get {search_re_items}.", )
                    )
                    agent.memory.save_context({}, {
                        agent.memory.add_memory_key: f"{name} searches for {item} in Amazon search engine.", }, )

                    observation = f"{name} is searching for {item} in Amazon search engine. The search engine returns {search_re_items}."
                    choice, action = agent.take_search_action(observation, self.now)

                    if "VIEW" in choice:
                        # safe model.
                        if len(search_re_items) == 0:
                            self.recsys.update_positive_by_id(agent_id, 'None', rd=self.round_cnt, type1=1, type2=1)
                            leave_ = True
                            continue
                        if not isinstance(action, int) or action > len(search_re_items):
                            action = 1

                        item_name = search_re_items[action - 1]
                        item_id = search_re_ids[action - 1]
                        item_detail = search_re_details[action - 1]

                        self.logger.info(f"{name} views {item_name}, in the search page.")
                        message.append(
                            Message(agent_id=agent_id, action="VIEW",
                                    content=f"{name} views {item_name}, in the search page.", )
                        )
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="VIEW",
                                    content=f"{name} views {item_name}, in the search page.", )
                        )
                        agent.memory.save_context({}, {
                            agent.memory.add_memory_key: f"{name} viewed {item_name}, in the search page.", }, )

                        agent.update_searched_history(item_name)
                        agent.update_viewed_history(item_name)
                        self.recsys.update_positive_by_id(agent_id, item_id, rd=self.round_cnt, type1=1, type2=1)
                        self.recsys.update_history_by_id(agent_id, item_id)

                        observation = (f"{name} is viewing the selected product." + item_detail)
                        choice, action = agent.take_sea_view_action(observation, self.now)

                        # self.recsys.add_train_data(agent_id, item_id, 1)
                        if "BUY" in choice or "CART" in choice:
                            self.recsys.add_train_data(agent_id, item_id, 1)
                        else:
                            self.recsys.add_train_data(agent_id, item_id, 0)

                        if "BUY" in choice:
                            self.logger.info(f"{name} buys {item_name}, in the search page.")
                            message.append(
                                Message(agent_id=agent_id, action="BUY",
                                        content=f"{name} buys {item_name}, in the search page.", ))
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="BUY",
                                        content=f"{name} buys {item_name}, in the search page.", ))
                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{agent.name} bought {item_name}, in the search page.", }, )

                            agent.update_purchased_history(item_name)
                            self.recsys.update_positive_by_id(agent_id, item_id, rd=self.round_cnt, type1=1, type2=3)

                            # getting review
                            observation = (f"{name} have bought the recommended product {item_name}." + item_detail)
                            rating, review = agent.generate_bought_review(observation, self.now, item_detail)
                            self.recsys.update_review(agent_id, item_id, rating, review, rd=self.round_cnt)

                            leave_ = True
                            leave = True
                            continue

                        elif "CART" in choice:
                            self.logger.info(f"{name} adds {item_name} to the shopping cart, in the search page.")
                            message.append(
                                Message(agent_id=agent_id, action="CART",
                                        content=f"{name} adds {item_name} to the shopping cart, in the search page.", ))
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="CART",
                                        content=f"{name} adds {item_name} to the shopping cart, in the search page.", ))

                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{name} adds {item_name} to the shopping cart, in the search page.", }, )

                            agent.update_cart({"itme_id": item_id, "item_name": item_name}, "add")
                            self.recsys.update_positive_by_id(agent_id, item_id, rd=self.round_cnt, type1=1, type2=2)

                            leave_ = True
                            continue

                        elif "RECOMMENDER" in choice:
                            self.logger.info(f"{name} leaves search engine and goes to recommender system.")
                            message.append(
                                Message(agent_id=agent_id, action="SEARCH VIEW GT RECOMMENDER",
                                        content=f"{name} leaves search engine and goes to recommender system.")
                            )
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="SEARCH VIEW GT RECOMMENDER",
                                        content=f"{name} leaves search engine and go to recommender system.")
                            )
                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{name} leaves search engine and goes to recommender system.", }, )
                            choice = "RECOMMENDER"
                            continue

                        elif "LEAVE" in choice:
                            self.logger.info(f"{name} leaves Amazon.")
                            message.append(Message(agent_id=agent_id, action="VIEW", content=f"{name} leaves Amazon."))
                            self.round_msg.append(Message(
                                agent_id=agent_id, action="VIEW", content=f"{name} leaves Amazon."))
                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{agent.name} leaves Amazon.", }, )
                            leave_ = True
                            leave = True
                            continue

                        else:
                            leave_ = True
                            continue

                    elif "NEXT" in choice:
                        self.logger.info(f"{name} searches for next item.")
                        message.append(Message(agent_id=agent_id, action="SEARCH",
                                               content=f"{name} searches for next item.", ))
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="SEARCH", content=f"{name} searches for next item.", ))
                        choice = "SEARCH"
                        continue

                    elif "RECOMMENDER" in choice:
                        self.logger.info(f"{name} leaves search engine and go to recommender system.")
                        message.append(
                            Message(
                                agent_id=agent_id, action="SEARCH VIEW GT RECOMMENDER",
                                content=f"{name} leaves search engine and go to recommender system."
                            )
                        )
                        self.round_msg.append(
                            Message(
                                agent_id=agent_id, action="SEARCH VIEW GT RECOMMENDER",
                                content=f"{name} search engine and go to recommender system."
                            )
                        )
                        choice = "RECOMMENDER"
                        continue

                    elif "LEAVE" in choice:
                        self.logger.info(f"{name} leaves Amazon.")
                        message.append(Message(agent_id=agent_id, action="SEARCH", content=f"{name} leaves Amazon.", ))
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="SEARCH", content=f"{name} leaves Amazon.", ))
                        leave_ = True
                        leave = True
                        continue

                    else:
                        leave_ = True
                        continue

                elif "SCART" in choice:
                    cart_item_ids, cart_items, item_details = agent.get_shopping_cart(self.data)
                    self.logger.info(f"{name} is browsing his/her shopping cart.")
                    message.append(
                        Message(agent_id=agent_id, action="SCART",
                                content=f"{name} is browsing his/her shopping cart.", ))
                    self.round_msg.append(
                        Message(agent_id=agent_id, action="SCART",
                                content=f"{name} is browsing his/her shopping cart.", ))

                    agent.memory.save_context({}, {
                        agent.memory.add_memory_key: f"{name} is browsing his/her shopping cart.", }, )

                    self.recsys.update_positive_by_id(agent_id, 'in', rd=self.round_cnt, type1=2, type2=0)

                    observation = (f"{name} is browsing his/her Amazon oline shopping cart. "
                                   + f"Products in {name}\'s cart are {cart_items}.")
                    choice, action = agent.take_cart_action(observation, self.now)

                    if "VIEW" in choice:
                        # safe model.
                        if len(cart_item_ids) == 0:
                            leave_ = True
                            continue
                        if not isinstance(action, int) or action > len(cart_item_ids):
                            action = random.randint(1, len(cart_item_ids))

                        item_name = cart_items[action - 1]
                        item_id = cart_item_ids[action - 1]
                        item_detail = item_details[action - 1]

                        self.logger.info(f"{name} views {item_name}, in his/her shopping cart.")
                        message.append(
                            Message(agent_id=agent_id, action="VIEW",
                                    content=f"{name} views {item_name}, in his/her shopping cart.", )
                        )
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="VIEW",
                                    content=f"{name} views {item_name}, in his/her shopping cart.", )
                        )
                        agent.memory.save_context({}, {
                            agent.memory.add_memory_key: f"{agent.name} viewed {item_name}, in his/her shopping cart.", }, )

                        observation = (f"{name} is viewing the selected product." + item_detail)
                        choice, action = agent.take_cart_view_action(observation, self.now)
                        self.recsys.update_positive_by_id(agent_id, item_id, rd=self.round_cnt, type1=2, type2=1)

                        if "BUY" in choice:
                            self.logger.info(f"{name} buys {item_name}")
                            message.append(
                                Message(agent_id=agent_id, action="BUY", content=f"{name} buys {item_name}.", ))
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="BUY", content=f"{name} buys {item_name}.", ))

                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{agent.name} bought {item_name}.", }, )

                            agent.update_purchased_history(item_name)
                            agent.update_cart({"itme_id": item_id, "item_name": item_name}, "de")
                            self.recsys.update_positive_by_id(agent_id, item_id, rd=self.round_cnt, type1=2, type2=3)

                            # getting review
                            observation = (f"{name} have bought the recommended product {item_name}." + item_detail)
                            rating, review = agent.generate_bought_review(observation, self.now, item_detail)
                            self.recsys.update_review(agent_id, item_id, rating, review, rd=self.round_cnt)

                            leave_ = True
                            leave = True
                            continue

                        elif "DELETE" in choice:
                            agent.update_cart({"itme_id": item_id, "item_name": item_name}, "de")
                            leave_ = True
                            continue

                        elif "SEARCH" in choice:
                            self.logger.info(f"{name} leaves recommender system and goes to the search engine.")
                            message.append(
                                Message(agent_id=agent_id, action="VIEW GT SEARCH",
                                        content=f"{name} leaves recommender system and goes to the search engine.")
                            )
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="VIEW GT SEARCH",
                                        content=f"{name} leaves recommender system and goes to the search engine.")
                            )
                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{name} leaves shopping cart and goes to the search engine.", },
                                                      )
                            choice = "SEARCH"
                            continue

                        elif "RECOMMENDER" in choice:
                            self.logger.info(f"{name} leaves shopping cart and go to recommender system.")
                            message.append(
                                Message(agent_id=agent_id, action="CART VIEW GT RECOMMENDER",
                                        content=f"{name} leaves shopping cart and go to recommender system."))
                            self.round_msg.append(
                                Message(
                                    agent_id=agent_id, action="CART VIEW GT RECOMMENDER",
                                    content=f"{name} leaves shopping cart and go to recommender system."
                                )
                            )
                            agent.memory.save_context({}, {
                                agent.memory.add_memory_key: f"{name} leaves shopping cart and goes to recommender system.", }, )

                            choice = "RECOMMENDER"
                            continue

                        elif "LEAVE" in choice:
                            self.logger.info(f"{name} leaves Amazon.")
                            message.append(Message(agent_id=agent_id, action="VIEW", content=f"{name} leaves Amazon."))
                            self.round_msg.append(
                                Message(agent_id=agent_id, action="VIEW", content=f"{name} leaves Amazon."))

                            agent.memory.save_context(
                                {}, {agent.memory.add_memory_key: f"{agent.name} leaves Amazon.", }, )
                            leave_ = True
                            leave = True
                            continue

                        else:
                            leave_ = True
                            continue

                    elif "SEARCH" in choice:
                        self.logger.info(f"{name} leaves shopping cart and goes to the search engine.")
                        message.append(
                            Message(agent_id=agent_id, action="CART GT SEARCH",
                                    content=f"{name} leaves shopping cart and goes to the search engine."))
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="CART GT SEARCH",
                                    content=f"{name} leaves shopping cart and goes to the search engine."))
                        choice = "SEARCH"
                        continue

                    elif "RECOMMENDER" in choice:
                        self.logger.info(f"{name} leaves shopping cart and go to recommender system.")
                        message.append(
                            Message(agent_id=agent_id, action="CART VIEW GT RECOMMENDER",
                                    content=f"{name} leaves shopping cart and go to recommender system."))
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="CART VIEW GT RECOMMENDER",
                                    content=f"{name} leaves shopping cart and go to recommender system."))
                        choice = "RECOMMENDER"
                        continue

                    elif "LEAVE" in choice:
                        self.logger.info(f"{name} leaves Amazon.")
                        message.append(Message(agent_id=agent_id, action="CART",
                                               content=f"{name} leaves Amazon from shopping cart.", ))
                        self.round_msg.append(
                            Message(agent_id=agent_id, action="CART",
                                    content=f"{name} leaves Amazon from shopping cart.", ))
                        leave_ = True
                        leave = True
                        continue

                    else:
                        leave_ = True
                        continue

                else:
                    self.logger.info(f"{name} does nothing.")
                    message.append(Message(agent_id=agent_id, action="NOTHING", content=f"{name} does nothing."))
                    self.round_msg.append(
                        Message(agent_id=agent_id, action="NOTHING", content=f"{name} does nothing.")
                    )
                    leave_ = True
                    leave = True

        if self.round_cnt in self.recsys.round_record[agent_id]:
            self.recsys.update_round_entropy(agent_id, self.round_cnt, self.data)
        return message

    def round(self):
        """ Run one step for all agents. """
        messages = []
        futures = []
        if self.config["execution_mode"] == "parallel":
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in tqdm(range(self.config["agent_num"])):
                    futures.append(executor.submit(self.one_step, i))
                    # time.sleep(10)
            for future in concurrent.futures.as_completed(futures):
                msgs = future.result()
                messages.append(msgs)
        else:
            for i in tqdm(range(self.config["agent_num"])):
                msgs = self.one_step(i)
                messages.append(msgs)

        return messages

    def create_agent(self, i) -> UserAgent:
        """
        Create an agent with the given id.
        """
        llm = utils.get_llm(logger=self.logger)
        now = 0
        agent_memory = RecAgentMemory(llm=llm, memory_retriever=self.create_new_memory_retriever(),
                                      verbose=False, reflection_threshold=10, now_=now)
        agent = UserAgent(
            id=i,
            name=self.data.users[i]["username"],
            age=self.data.users[i]["age"],
            gender=self.data.users[i]["gender"],
            traits=utils.get_string_from_list(self.data.users[i]["traits"]),
            status=self.data.users[i]["status"],
            interest=utils.get_string_from_list(self.data.users[i]["interests"]),
            feature=utils.get_string_from_list(self.data.users[i]["features_used"]),
            income=self.data.users[i]["annual_income"],
            TIPI=self.data.users[i]["TIPI"],
            memory_retriever=self.create_new_memory_retriever(),
            llm=llm,
            memory=agent_memory,
        )
        # observations = self.data.users[i]["observations"].strip(".").split(".")
        # for observation in observations:
        #     agent.memory.add_memory(observation, now=self.now)
        return agent

    def agent_creation(self):
        """
        Create agents in parallel
        """
        agents = {}
        agent_num = int(self.config["agent_num"])
        if self.active_method == "random":
            active_probs = [self.config["active_prob"]] * agent_num
        else:
            # active_probs = np.random.pareto(self.config["active_prob"] * 10, agent_num)
            # active_probs = np.sort(active_probs / active_probs.max())
            # print(active_probs)
            active_probs = []
            x = [60, 22, 12, 8, 6, 5, 4, 4, 3, 2, 2, 2, 2, 1, 1]
            for i, v in enumerate(x):
                active_probs.extend([(i + 1) / 15] * v)


        if self.config["execution_mode"] == "parallel":
            futures = []
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(agent_num):
                    futures.append(executor.submit(self.create_agent, i))
                for future in tqdm(concurrent.futures.as_completed(futures)):
                    agent = future.result()
                    agent.active_prob = active_probs[agent.id]
                    agents[agent.id] = agent
            end_time = time.time()
            self.logger.info(
                f"Time for creating {agent_num} agents: {end_time - start_time}"
            )
        else:
            for i in tqdm(range(agent_num)):
                agent = self.create_agent(i)
                agent.active_prob = active_probs[agent.id]
                agents[agent.id] = agent

        return agents


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, default='messages.json', help="Path to output file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="messages.log", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    # create config
    config = CfgNode(new_allowed=True)
    output_file = os.path.join("output/message", args.output_file)
    config = utils.add_variable_to_config(config, "output_file", output_file)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config.merge_from_file(args.config_file)
    logger.info(f"\n{config}")

    if config["simulator_restore_file_name"]:
        restore_path = os.path.join(
            config["simulator_dir"], config["simulator_restore_file_name"]
        )
        simulator = Simulator.restore(restore_path, config, logger)
        logger.info(f"Successfully Restore simulator from the file <{restore_path}>\n")
        logger.info(f"Start from the round {simulator.round_cnt + 1}\n")
    else:
        simulator = Simulator(config, logger)
        simulator.load_simulator()

    messages = []
    for i in tqdm(range(simulator.round_cnt + 1, config["round"] + 1)):
        simulator.round_cnt = simulator.round_cnt + 1
        simulator.logger.info(f"Round {simulator.round_cnt}")
        simulator.active_agents.clear()
        message_ = simulator.round()
        messages.append(message_)
        with open(config["output_file"], "w") as file:
            json.dump(messages, file, default=lambda o: o.__dict__, indent=4)
        simulator.recsys.save_interaction()

        if i % 1 == 0:
            simulator.recsys.train()
        if i % 10 == 0:
            simulator.save(os.path.join(config["simulator_dir"]))


if __name__ == "__main__":
    main()
