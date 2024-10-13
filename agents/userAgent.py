import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from langchain_experimental.generative_agents.generative_agent import GenerativeAgent
from langchain.prompts import PromptTemplate
from langchain_core.memory import BaseMemory


class UserAgent(GenerativeAgent):
    id: int
    """The agent's unique identifier"""

    name = str
    """The agent's name"""

    age: int
    """The agent's age"""

    gender: str
    """The agent's gender"""

    traits: str
    """The agent's traits"""

    interest: str
    """The agent's interest"""

    feature: str
    """The agent's feature"""

    income: str
    """The agent's annual income"""

    TIPI: dict = {}
    """The agent's TIPI feature"""

    viewed_history: List[str] = []
    """The agent's history of viewed products"""

    purchased_history: List[str] = []
    """The agent's history of purchased products"""

    searched_history: List[str] = []
    """The agent's history of searched products"""

    shopping_cart: List[dict] = []
    """The agent's shopping cart list"""

    active_prob: float = 0.5
    """The probability of the agent being active"""

    no_action_round: int = 0
    """The number of rounds that the agent has not taken action"""

    memory: BaseMemory
    """The memory module in UserAgent."""

    role: str = "agent"

    failed_num: int = 0
    """The number of failed actions that the agent has taken"""

    # def __lt__(self, other: "UserAgent"):
    #     return self.event.end_time < other.event.end_time

    last_refreshed: datetime = None

    def get_active_prob(self, method) -> float:
        if method == "marginal":
            return self.active_prob * (self.no_action_round + 1)
        else:
            return self.active_prob

    def get_summary_(self, observation: str = None, ) -> str:
        """Return a descriptive summary of the agent."""
        prompt = PromptTemplate.from_template(
            "Given the following observation about {agent_name}: '{observation}', please summarize the relevant details from his/her profile. His profile information is as follows:\n"
            + "Name: {agent_name}\n"
            + "Age: {agent_age}\n"
            + "Gender:{agent_gender}\n"
            + "Traits: {agent_traits}\n"
            + "Status: {agent_status}\n"
            + "Interest: {agent_interest}\n"
            + "Feature: {agent_feature}\n"
            + "income: {agent_income}$ \n"
            + "The TEN ITEM PERSONALITY MEASURE is statement as follows:\n"
            + 'Here are a number of personality traits that may or may not apply to you. Please write a number next to each statement to indicate the extent to which you agree or disagree with that statement. You should rate the extent to which the pair of traits applies to you, even if one characteristic applies more strongly than the other.'
            + 'Disagree strongly: 1; Disagree moderately: 2; Disagree a little: 3; Neither agree nor disagree: 4; Agree a little: 5; Agree moderately: 6; Agree strongly: 7.'
            + 'I see myself as: Q1. _____ Extraverted, enthusiastic. Q2. _____ Critical, quarrelsome. Q3. _____ Dependable, self-disciplined. Q4. _____ Anxious, easily upset. Q5. _____ Open to new experiences, complex. Q6. _____ Reserved, quiet. Q7. _____ Sympathetic, warm. Q8. _____ Disorganized, careless. Q9. _____ Calm, emotionally stable. Q10. _____ Conventional, uncreative.\n'
            + "{agent_name} answers: {agent_TIPI}"
            + "Please avoid repeating the observation in the summary.\nSummary:"
        )
        kwargs: Dict[str, Any] = dict(
            observation=observation,
            agent_name=self.name,
            agent_age=self.age,
            agent_gender=self.gender,
            agent_traits=self.traits,
            agent_status=self.status,
            agent_interest=self.interest,
            agent_feature=self.feature,
            agent_TIPI=self.TIPI,
            agent_income=self.income,
        )
        result = self.chain(prompt=prompt).run(**kwargs).strip()
        age = self.age if self.age is not None else "N/A"
        gender = self.gender if self.gender is not None else "N/A"
        return f"Name: {self.name} (age: {age}, gender: {gender})" + f"\n{result}"

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            """
{q1}?
Context from memory:
{most_recent_memories}
Relevant context: 
"""
        )
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        most_recent_memories = self.memory.load_memory_variables({'observation': observation})['most_recent_memories']
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2], observation=observation,
                                             most_recent_memories=most_recent_memories).strip()

    def _generate_reaction(
            self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name} recently viewed {viewed_history}."
            + "\n{agent_name} recently added {shopping_cart} to the shopping cart."
            + "\n{agent_name} recently purchased {purchased_history}."
            + "\n{agent_name} recently searched {searched_history} on search engines."
            + "\nOther than that {agent_name} doesn't know any other products."
            + "\nObservation: {observation}."
            + "\nRelevant memories of {agent_name}: {relevant_memories}."
            + "\nAll occurrences of products' names should be enclosed with <>"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary_(observation=observation)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (datetime.now().strftime("%B %d, %Y, %I:%M %p")
                            if now is None
                            else now.strftime("%B %d, %Y, %I:%M %p"))
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
            viewed_history=(self.viewed_history[-1:-10] if len(self.viewed_history) > 10 else self.viewed_history),
            shopping_cart=(self.shopping_cart[-1:-10] if len(self.shopping_cart) > 10 else self.shopping_cart),
            purchased_history=(
                self.purchased_history[-1:-10] if len(self.purchased_history) > 10 else self.purchased_history),
            searched_history=(
                self.searched_history[-1:-10] if len(self.searched_history) > 10 else self.searched_history),
        )
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def take_action(self, now) -> Tuple[str, str]:
        """Take one of the actions below.
        (1) Enter the Recommender.
        (2) Enter the Search Engines.
        (3) Enter the shopping cart.
        (4) Do Nothing.
        """
        call_to_action_template = (
            "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
            "First, consider whether you have a specific or vague goal for buying some products interests? "
            "If so, choose to enter the search engine, and write:\n [SEARCH]:: {agent_name} enters the Search Engine. \n"
            "else if you want to explore amazon's offerings provided by Amazon recommender, choose to view recommendations and, write:\n [RECOMMENDER]:: {agent_name} enters the Recommender System. \n"
            "else if you want to view or purchase items that have been added to your cart, write:\n [SCART]:: {agent_name} enters the shopping cart. \n"
            "Note that at this point, you may feel tired from browsing Amazon for a long time, or the content provided by Amazon is not novel enough. If so, you can also choose to leave Amazon and give your reasons for leaving, write:\n [NOTHING]:: {agent_name} does nothing, the reason is [here is your reason]. "
        )
        observation = (
                "You're {agent_name}, now. you must take only ONE of the actions below:"
                + f"\n(1) Enter the Recommender System. If so, you will be recommended some products, from which you can view details of the products, add products to the shopping cart, or purchase the products."
                + f"\n(2) Enter the Search Engines. If so, you can search for any products he want to know by himself at this time."
                + f"\n(2) Enter the shopping cart. If so, you can view product list of his/her shopping cart and buy some product added to the shopping cart."
                + f"\n(3) Do Nothing."
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = "[LEAVE]:: " + str(self.name) + " does nothing."
        for item in full_result.strip().split("\n")[::-1]:
            if item.find("::") != -1:
                result = item
                break
        try:
            choice, action = result.split("::")
        except:
            self.failed_num += 1
            choice, action = "[LEAVE]:: does nothing.".split("::")

        # result = full_result.strip().split("\n")[0]
        # choice = result.split("::")[0]

        self.memory.save_context({}, {self.memory.add_memory_key: f"{self.name} take action: {action}", }, )
        return choice, action

    def take_recommender_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) View details page of the recommended items.
        (2) Next page.
        (3) Search items.
        (4) Leave Amazon.
        """
        call_to_action_template = (
                "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
                "you must choose one of the four actions below:\n"
                "(1) View ONLY ONE product's details page from the list returned by the recommender system.\n"
                "(2) See the next page.\n"
                "(3) Search for a specific item.\n"
                "(4) Leave Amazon."
                + "\n If you want to view the details page of A product from the recommended list that match {agent_name}'s interests, write:\n[VIEW]:: Index of the product starting from 1 (e.g., [VIEW]:: 3)"
                + "\n else if there is no item you interests, and want to see the next page of the recommendation, write:\n[NEXT]:: {agent_name} views the next page."
                + "\n else if you want to search for a specific item, write:\n[SEARCH]:: single, specific words about the items to search for."
                + "\n else, you may feel tired from browsing Amazon for a long time, or the content provided by Amazon is not novel enough. If so, you can also choose to leave Amazon and give your reasons for leaving, write:\n[LEAVE]:: {agent_name} leaves Amazon, the reason is [here is your reason]. "
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = "[LEAVE]:: " + str(self.name) + " leaves Amazon."
        for item in full_result.strip().split("\n")[::-1]:
            if item.find("::") != -1:
                result = item
                break
        try:
            choice, action = result.split("::")
        except:
            self.failed_num += 1
            choice, action = "[LEAVE]:: leaves Amazon.".split("::")

        choice = choice.strip()
        match = re.search(r'(\d+)', action.strip())
        if match:
            num = int(match.group(1))
            if 1 <= num <= 5:
                action = num
            else:
                action = 1

        self.memory.save_context({}, {
            self.memory.add_memory_key: f"{self.name} take action: {result}, from the recommendation page.", }, )
        return choice, action

    def take_rec_view_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the five actions below.
           (1) Add current product to the shopping cart.
           (2) Buy the current product.
           (3) Return to recommendation page.
           (4) Go to the search engine.
           (5) Leave Amazon.
        """
        call_to_action_template = (
                "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
                "You must choose one of the four actions below:\n"
                "(1) Add the current product to the shopping cart.\n"
                "(2) Buy the current product outright.\n"
                "(3) Return to the recommendation page.\n"
                "(4) Search for a specific item.\n"
                "(5) Leave Amazon. \n"
                + "If the current product match your interests and needs very well, and the price is right, you can buy it outright, and write:\n[BUY]:: {agent_name} bought the current product.\n"
                + "else if the current product match your interests or needs, but you don't want to buy it now, you can add it to your shopping cart, and write:\n[CART]:: {agent_name} add the current product to the shopping cart.\n"
                + "else if you want to know a particular product now, you should choose to search for it on the Amazon search engine, write:\n[SEARCH]:: single, specific words about the items to search for.\n"
                + "else if you want to go to recommendation page to explore other products, still, write:\n[RECOMMENDER]:: {agent_name} go to the Recommender System. \n"
                + "else, you may feel tired from browsing Amazon for a long time, or the content provided by Amazon is not novel enough. If so, you can also choose to leave Amazon and give your reasons for leaving, write:\n[LEAVE]:: {agent_name} leaves Amazon, the reason is [here is your reason]. "
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = "[LEAVE]::" + str(self.name) + "leaves Amazon."
        for item in full_result.strip().split("\n")[::-1]:
            if item.find("::") != -1:
                result = item
                break
        try:
            choice, action = result.split("::")
        except:
            self.failed_num += 1
            choice, action = "[LEAVE]:: leaves Amazon.".split("::")

        choice = choice.strip()
        return choice, action

    def take_cart_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
            (1) View details page of one of the items.
            (2) Go to recommendation page.
            (3) Search items.
            (4) Leave Amazon.
        """
        call_to_action_template = (
                "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
                "You must choose one of the four actions below:\n"
                "(1) View ONLY ONE product's details page from the cart list.\n"
                "(2) Search for another item.\n"
                "(3) Go to the recommender system.\n"
                "(4) Leave Amazon.\n"
                + "If you want to view the details page of a product in your shopping cart, write:\n[VIEW]:: Index of the product starting from 1 (e.g., [VIEW]:: 3) \n"
                + "else if you want to know some products, you can search for them on the Amazon search engine, and write:\n[SEARCH]:: {agent_name} is going to search for an item. \n"
                + "else if you want to go to the Amazon recommender system, write:\n[RECOMMENDER]:: {agent_name} is going to Amazon recommender system. \n"
                + "else, you may feel tired from browsing Amazon for a long time, or the content provided by Amazon is not novel enough. If so, you can also choose to leave Amazon and give your reasons for leaving, write:\n[LEAVE]:: {agent_name} leaves Amazon, the reason is [here is your reason]. "
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = "[LEAVE]::" + str(self.name) + "leaves Amazon."
        for item in full_result.strip().split("\n")[::-1]:
            if item.find("::") != -1:
                result = item
                break
        try:
            choice, action = result.split("::")
        except:
            # print('\n'+result+'\n')
            self.failed_num += 1
            choice, action = "[LEAVE]:: leaves Amazon.".split("::")

        # choice, action = result.split("::")
        choice = choice.strip()
        match = re.search(r'(\d+)', action)
        if match:
            action = int(match.group(1))

        self.memory.save_context({}, {
            self.memory.add_memory_key: f"{self.name} took action: {full_result}, from the shopping cart page.", }, )
        return choice, action

    def take_cart_view_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
           (1) Buy the current product.
           (2) Return to recommendation page.
           (3) Go to the search engine.
           (4) Leave Amazon.
        """
        call_to_action_template = (
                "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
                "You must choose one of the four actions below:\n"
                "(1) Buy the current product outright.\n"
                "(2) Delete the current product from shopping cart.\n"
                "(3) Return to the recommendation page.\n"
                "(4) Search for a specific item.\n"
                "(5) Leave Amazon.\n"
                + "If you finally decides to buy the current product which has been added to your shopping cart, write:\n[BUY]:: {agent_name} bought the current product.\n"
                + "else if you doesn't consider to buy the current product in the future and decides to delete it from the shopping cart, write:\n[DELETE]:: {agent_name} delete the current product.\n"
                + "else if you wants to know an another particular product, you choose to search for it on the Amazon search engine, write:\n[SEARCH]:: single, specific words about the items to search for.\n"
                + "else if you wants go to recommendation page to explore other products, write:\n[RECOMMENDER]:: {agent_name} go to the Recommender System.\n"
                + "else, you may feel tired from browsing Amazon for a long time, or the content provided by Amazon is not novel enough. If so, you can also choose to leave Amazon and give your reasons for leaving, write:\n[LEAVE]:: {agent_name} leaves Amazon, the reason is [here is your reason]. "
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = "[LEAVE]::" + str(self.name) + "leaves Amazon."
        for item in full_result.strip().split("\n")[::-1]:
            if item.find("::") != -1:
                result = item
                break
        try:
            choice, action = result.split("::")
        except:
            self.failed_num += 1
            choice, action = "[LEAVE]:: leaves Amazon.".split("::")

        choice = choice.strip()
        return choice, action

    def take_search_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) View details page of the returned items.
        (2) Next search.
        (3) Go to recommendation page.
        (4) Leave Amazon.
        """
        call_to_action_template = (
                "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
                "You must choose one of the five actions below:\n"
                "(1) View ONLY ONE product's details page from the list returned by the recommender system.\n"
                "(2) Search for another item.\n"
                "(3) Go to the recommender system.\n"
                "(4) Leave Amazon.\n"
                + "If you want to view the details page of a product from the search engine returns list that match {agent_name}'s interests, write:\n[VIEW]:: Index of the product starting from 1 (e.g., [VIEW]:: 3)"
                + "else if you want to search for another item, write:\n[NEXT]:: {agent_name} is going to search for another item."
                + "else if you wants go to the recommender system, write:\n[RECOMMENDER]:: {agent_name} is going to Amazon recommender system."
                + "else, you may feel tired from browsing Amazon for a long time, or the content provided by Amazon is not novel enough. If so, you can also choose to leave Amazon if you want to do nothing now, and give your reasons for leaving, write:\n[LEAVE]:: {agent_name} leaves Amazon, the reason is [here is your reason]. "
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = "[LEAVE]::" + str(self.name) + "leaves Amazon."
        for item in full_result.strip().split("\n")[::-1]:
            if item.find("::") != -1:
                result = item
                break
        try:
            choice, action = result.split("::")
        except:
            self.failed_num += 1
            choice, action = "[LEAVE]:: leaves Amazon.".split("::")

        choice = choice.strip()
        match = re.search(r'(\d+)', action)
        if match:
            action = int(match.group(1))

        self.memory.save_context({}, {
            self.memory.add_memory_key: f"{self.name} took action: {result}, from the search engine page.", }, )
        return choice, action

    def take_sea_view_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the five actions below.
           (1) Add current product to the shopping cart.
           (2) Buy the current product.
           (3) Return to recommendation page.
           (4) Go to the search engine.
           (5) Leave Amazon.
        """
        call_to_action_template = (
                "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
                "You must choose one of the four actions below:\n"
                "(1) Add the current product to the shopping cart.\n"
                "(2) Buy the current product outright.\n"
                "(3) Return to the recommendation page.\n"
                "(4) Search for a specific item.\n"
                "(5) Leave Amazon. \n"
                + "If the current product match your interests or needs very well and the price is right, you can buy it outright, and write:\n[BUY]:: {agent_name} bought the current product.\n"
                + "else if the current product match your interests or needs, but you don't want to buy it now, you can add it to your shopping cart, and write:\n[CART]:: {agent_name} add the current product to the shopping cart.\n"
                + "else if you want to know a particular product now, still, you should choose to search for it on the Amazon search engine, write:\n[SEARCH]:: single, specific words about the items to search for.\n"
                + "else if you want to go to recommendation page to explore other products, write:\n[RECOMMENDER]:: {agent_name} go to the Recommender System. \n"
                + "else, you may feel tired from browsing Amazon for a long time, or the content provided by Amazon is not novel enough. If so, you can also choose to leave Amazon and give your reasons for leaving, write:\n[LEAVE]:: {agent_name} leaves Amazon, the reason is [here is your reason]. "
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = "[LEAVE]::" + str(self.name) + "leaves Amazon."
        for item in full_result.strip().split("\n")[::-1]:
            if item.find("::") != -1:
                result = item
                break
        try:
            choice, action = result.split("::")
        except:
            self.failed_num += 1
            choice, action = "[LEAVE]:: leaves Amazon.".split("::")

        choice = choice.strip()
        return choice, action

    def search_item(self, observation, now) -> str:
        """Search item by the item name."""
        call_to_action_template = (
                "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
                "What products or item would you interested in and want to search for it in the Amazon this time? "
                + "Respond only one item's you want to search, for example: "
                + "\n"
                + "<Science Books>"
                + "\n\n"
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)
        full_result = self.get_item_from_stats(full_result)

        # result = full_result
        # for item in full_result.strip().split("\n")[::-1]:
        #     if item.find("<") != -1 and item.find(">") != -1:
        #         tt = str.maketrans('\n', '\n', '<>')
        #         result = item.translate(tt)
        #         break

        return full_result

    def generate_bought_review(self, observation: str, now, detail) -> Tuple[int, str]:
        """Generate rating score and review for the bought product."""
        call_to_action_template = (
                "Act as {agent_name} according to the information of {agent_name} provided above. And now you are {agent_name}. \n"
                "You just bought a products: " + detail.translate(str.maketrans('', '', '{}')) + '\n'
                + "Considering your preferences, purchase history, and historical reviews, how would you rate and evaluate your the most recent purchase, and why? Answer according to the following rules:\n"
                + "(1) RATING, the rating score indicates your satisfaction with this purchase. Very Critically rate this product on a scale from 1 to 5, where 1 means you really dislike it and 5 means you really like it.\n"
                + "(2) REVIEW, brief comments made by you after purchasing the product, which can include various aspects such as product quality and use experience.\n"
                + "The shorter the answer, the better, the following sample format must be followed:\n\n"
                + "[RATING]::3\n"
                + "[REVIEW]::Some feelings and comments, in one line."
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        rating = 0
        review = ''
        for res in full_result.strip().split("\n")[::-1]:
            if res.find("[RATING]::") != -1:
                result = res.strip().split("::")[-1]
                match = re.search(r'(\d+)', result.strip())
                if match:
                    num = int(match.group(1))
                    if 1 <= num <= 5:
                        rating = num
                    elif num >= 5:
                        rating = 5
            if res.find("[REVIEW]::") != -1:
                review = res.strip().split("::")[-1]

        return rating, review

    def update_cart(self, item: dict, tp: str):
        """Update shopping cart."""
        if tp == "add":
            self.shopping_cart.extend([item])
        elif tp == "de":
            for pd in self.shopping_cart:
                if pd["itme_id"] == item["itme_id"]:
                    self.shopping_cart.remove(pd)

    def get_shopping_cart(self, data):
        item_ids = [item["itme_id"] for item in self.shopping_cart]
        item_names = data.get_item_names(item_ids)
        item_details = data.get_item_description_by_id(item_ids)
        return item_ids, item_names, item_details

    def update_viewed_history(self, item, now=None):
        """Update viewed history. If the number of items in the history achieves the BUFFERSIZE, delete the oldest
        item."""
        self.viewed_history.append(item)
        # if len(self.viewed_history) > self.BUFFERSIZE:
        #     self.viewed_history = self.viewed_history[-self.BUFFERSIZE:]

    def update_purchased_history(self, item, now=None):
        """Update history by the items bought. If the number of items in the history achieves the BUFFERSIZE,
        delete the oldest item."""
        self.purchased_history.append(item)
        # self.purchased_history.extend(items)
        # if len(self.purchased_history) > self.BUFFERSIZE:
        #     self.purchased_history = self.purchased_history[-self.BUFFERSIZE:]

    def update_searched_history(self, item, now=None):
        """Update history by the items searched. If the number of items in the history achieves the BUFFERSIZE,
        delete the oldest item."""
        self.searched_history.append(item)
        # self.searched_history.extend(items)
        # if len(self.searched_history) > self.BUFFERSIZE:
        #     self.searched_history = self.searched_history[-self.BUFFERSIZE:]

    def get_item_from_stats(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "Respond only one item's name this one want to search, for example: "
            + "\n"
            + "<Science Books>"
            + "\n\n"
        )
        return self.chain(prompt).run(observation=observation).strip()
