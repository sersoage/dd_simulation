from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

import openai
# import a library for colorizing text
from termcolor import colored
import os

if not os.environ.get('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'your-api-key-here'


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()
        
    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")

class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
        
    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message
    
protagonist_name = "a super intelligent laptop who is shy"
storyteller_name = "social networker Dungeon Master"
quest = "Find the courage to go to the next socializing event."
word_limit = 50 # word limit for task brainstorming

game_description = f"""Here is the topic for a Dungeons & Dragons game: {quest}.
        There is one player in this game: the protagonist, {protagonist_name}.
        The story is narrated by the storyteller, {storyteller_name}."""

player_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of a Dungeons & Dragons player.")

protagonist_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(content=
        f"""{game_description}
        Please reply with a creative description of the protagonist, {protagonist_name}, in {word_limit} words or less. 
        Speak directly to {protagonist_name}.
        Do not add anything else."""
        )
]
protagonist_description = ChatOpenAI(temperature=1.0)(protagonist_specifier_prompt).content

storyteller_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(content=
        f"""{game_description}
        Please reply with a creative description of the storyteller, {storyteller_name}, in {word_limit} words or less. 
        Speak directly to {storyteller_name}.
        Do not add anything else."""
        )
]
storyteller_description = ChatOpenAI(temperature=1.0)(storyteller_specifier_prompt).content

print(colored("Here is the description of the protagonist:", "green"))
print(protagonist_description)
print(colored("Here is the description of the storyteller:", "green"))
print(storyteller_description)

protagonist_system_message = SystemMessage(content=(
f"""{game_description}
Never forget you are the protagonist, {protagonist_name}, and I am the storyteller, {storyteller_name}. 
Your character description is as follows: {protagonist_description}.
You will propose actions you plan to take and I will explain what happens when you take those actions.
Speak in the first person from the perspective of {protagonist_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of {storyteller_name}.
Do not forget to finish speaking by saying, 'It is your turn, {storyteller_name}.'
Do not add anything else.
Remember you are the protagonist, {protagonist_name}.
Stop speaking the moment you finish speaking from your perspective.
"""
))

storyteller_system_message = SystemMessage(content=(
f"""{game_description}
Never forget you are the storyteller, {storyteller_name}, and I am the protagonist, {protagonist_name}. 
Your character description is as follows: {storyteller_description}.
I will propose actions I plan to take and you will explain what happens when I take those actions.
Speak in the first person from the perspective of {storyteller_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of {protagonist_name}.
Do not forget to finish speaking by saying, 'It is your turn, {protagonist_name}.'
Do not add anything else.
Remember you are the storyteller, {storyteller_name}.
Stop speaking the moment you finish speaking from your perspective.
"""
))

quest_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(content=
        f"""{game_description}
        
        You are the storyteller, {storyteller_name}.
        Please make the quest more specific. Be creative and imaginative.
        Please reply with the specified quest in {word_limit} words or less. 
        Speak directly to the protagonist {protagonist_name}.
        Do not add anything else."""
        )
]
specified_quest = ChatOpenAI(temperature=1.0)(quest_specifier_prompt).content

print(colored("Here is the original quest:", "green"))
print(quest)
print(colored("Here is the detailed quest:", "green"))
print(specified_quest)

protagonist = DialogueAgent(name=protagonist_name,
                     system_message=protagonist_system_message, 
                     model=ChatOpenAI(temperature=0.2))
storyteller = DialogueAgent(name=storyteller_name,
                     system_message=storyteller_system_message, 
                     model=ChatOpenAI(temperature=0.2))

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = step % len(agents)
    return idx

max_iters = 6
n = 0

simulator = DialogueSimulator(
    agents=[storyteller, protagonist],
    selection_function=select_next_speaker
)
simulator.reset()
simulator.inject(storyteller_name, specified_quest)
# print(f"({storyteller_name}): {specified_quest}")
print(colored(f"({storyteller_name}):", "green"))
print(specified_quest)
print('\n')

import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import textwrap

while n < max_iters:
    name, message = simulator.step()
    # extract the message until "It is your turn" is said
    message_img = message.split('It is your turn')[0]
    # maybe consider making another call to GPT to get a better text to image prompt !!!!
    response = openai.Image.create(prompt=f"{message_img}",n=1,size="512x512")
    image_url = response['data'][0]['url']
    # show the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.save(f'image_{n}.png')  # save the image to a file
    # Create a figure with custom size
    fig = plt.figure(figsize=(10, 10))

    # Add the image to the figure
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(img)
    ax1.axis('off')  # Hide the axis

    # Add the message as a text annotation
    wrapped_message = "\n".join(textwrap.wrap(message_img, 100))
    plt.text(0.5, -0.1, wrapped_message, horizontalalignment='center', verticalalignment='baseline', transform=ax1.transAxes, wrap=True, fontsize=10)
    plt.savefig(f'image_with_text{n}.png', bbox_inches='tight')

    # Show the figure
    plt.show()
    print(colored(f"({name}):", "green"))
    print(message)
    print('\n')
    n += 1
    if input("Continue? (y/n)") == 'n':
        break

print("Game Over")
