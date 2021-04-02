import gzip
from typing import Tuple, List

import argparse
from collections import OrderedDict, deque, namedtuple
import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F

from mechanics.Cards import Deck_of_Cards_Class
from mechanics.Square_Grid_Movement import Square_Grid_Movement_Class
from model.entity_combination import Attention_Model
from model.entity_duplication import Duplicate_Entities_Model
from model.entity_generation import ManyToOneEncoder
from model.entity_initialization import Initializer_Model

from pytorch_lightning import LightningModule, Trainer
import time



# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state', 'length', 'metadata_file'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque()

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def pop(self) -> Experience:
        """
        Pop experience from replay buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        return self.buffer.popleft()

    def sample(self, batch_size: int, max_seq_length=100) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states, lengths, metadata = zip(*[self.buffer[idx] for idx in indices])

        # states = []
        # next_states = []
        # for state_path, next_state_path in zip(state_paths, next_state_paths):
        #     state = torch.load(state_path)
        #     next_state = torch.load(next_state_path)
        #     L, W = state.shape
        #     nL, nW = next_state.shape
        #
        #     # Pad the states
        #     padded_state = torch.zeros(max_seq_length, W)
        #     padded_next_state = torch.zeros(max_seq_length, W)
        #     padded_state[:L,:] = state
        #     padded_next_state[:nL,:] = next_state
        #
        #     states.append(padded_state)
        #     next_states.append(padded_next_state)

        return (torch.stack(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.bool), torch.stack(next_states), torch.tensor(lengths))

class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 50) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states, lengths = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i], lengths[i]


class DQNLightning(LightningModule):
    """ Basic DQN Model """

    def __init__(self,
                 replay_size,
                 warm_start_steps: int,
                 gamma: float,
                 eps_last_frame: int,
                 sync_rate,
                 lr: float,
                 episode_length,
                 batch_size,
                 metadata_path='/home/jamison/projects/game_generation/meta_data',
                 data_path='/home/jamison/projects/game_generation/data',
                 **kwargs) -> None:
        super().__init__()
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.eps_last_frame = eps_last_frame
        self.sync_rate = sync_rate
        self.lr = lr
        self.episode_length = episode_length
        self.batch_size = batch_size

        # Get the basic mechanic dicts and lists
        mechanic_list = ["Square-Grid Movement", "Deck-of-Cards"]  # "Betting"]
        mechanic_types = {
            "Square-Grid Movement": 1,
            "Betting": 2,
        }
        mechanic_dicts, mechanic_objs = {}, {}
        if "Square-Grid Movement" in mechanic_list:
            Square_Class = Square_Grid_Movement_Class()
            mechanic_dicts[1] = Square_Class.get_mechanic_dict()  # "Square-Grid Movement"
            mechanic_objs[1] = Square_Class  # "Square-Grid Movement"
        if "Deck-of-Cards" in mechanic_list:
            Deck_Class = Deck_of_Cards_Class()
            mechanic_dicts[2] = Deck_Class.get_mechanic_dict()  # "Deck-of-Cards"
            mechanic_objs[2] = Deck_Class  # "Deck-of-Cards"

        # Initialize the networks
        self.generation_net = ManyToOneEncoder()
        self.target_generation_net = ManyToOneEncoder()

        self.combination_net = Attention_Model()
        self.duplication_net = Duplicate_Entities_Model(mechanic_types, mechanic_dicts)
        self.initialization_net = Initializer_Model()

        self.buffer = ReplayBuffer(self.replay_size)
        self.total_reward = 0
        self.episode_reward = 0
        self.min_buffer_length = 1000

        # Parameters that deal with data
        self.files_in_buffer = set()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.average_value = -1.0

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Passes in a state `x` through the network and gets the `q_values` of each action as an output
        Args:
            x: environment state
        Returns:
            q values
        """
        output = None
        if mode == "generation":
            output = self.generation_net(x)

        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str = "generation") -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states, lengths = batch

        if mode == "generation":
            state_action_values = self.generation_net(states, device=self.device)["logits"].gather(1, actions.unsqueeze(-1)).squeeze(-1)

        else:
            raise NotImplementedError

        with torch.no_grad():
            next_state_values = self.target_generation_net(next_states)["logits"].max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return F.smooth_l1_loss(state_action_values, expected_state_action_values)

    def update_replay_buffer(self, max_sequence_length=100, max_files_to_load=10000):
        all_files = set()
        for file in os.listdir(self.metadata_path):
            fullpath = os.path.join(self.metadata_path, file)
            if os.path.isfile(fullpath):
                all_files.add(fullpath)

        # Get files in buffer that are not in the saved location
        new_files = all_files - self.files_in_buffer

        if new_files == 0:
            return self.average_value

        file_counter = 0
        stop_adding_files = False
        files_to_remove = set()
        values = []
        for metadata_file in new_files:
            if not stop_adding_files:
                for item in self.parse(metadata_file):
                    state = torch.load(item["state_path"])
                    next_state = torch.load(item["next_state_path"])
                    L, W = state.shape
                    nL, nW = next_state.shape

                    # Pad the states
                    padded_state = torch.zeros(max_sequence_length, W)
                    padded_next_state = torch.zeros(max_sequence_length, W)
                    padded_state[:L,:] = state
                    padded_next_state[:nL,:] = next_state

                    # Add to experience tuple
                    experience = Experience(padded_state,
                               item["action"],
                               item["reward"],
                               bool(item["done"]),
                               padded_next_state,
                               item["length"],
                               metadata_file)

                    # Add experience to replay buffer
                    self.buffer.append(experience)
                    # Remove from replay buffer if we have exceeded the capacity
                    if len(self.buffer) > self.buffer.capacity:
                        removed_experience = self.buffer.pop()
                        files_to_remove.add(removed_experience.metadata_file)

                    # Append to values list
                    values.append(item["value"])

                # Add the metadata_file to files in buffer
                self.files_in_buffer.add(metadata_file)
            else:
                # Remove extra files we don't get around to adding as well
                files_to_remove.add(metadata_file)


            # If we have added too many files, skip
            file_counter += 1
            if file_counter >= max_files_to_load:
                stop_adding_files = True

        # Get rid of the extra data we are not using
        self.delete_extra_files(files_to_remove)

        if len(values) > 0:
            return np.mean(values)
        else:
            return self.average_value

    def delete_extra_files(self, files_to_remove):
        for file in files_to_remove:
            try:
                for item in self.parse(file):
                    os.remove(item["state_path"])
                    os.remove(item["next_state_path"])
                os.remove(file)
            except FileNotFoundError:
                # If the file is not found, then that means we already deleted it
                pass

    def parse(self, path):
        g = gzip.open(path, 'r')
        for l in g:
            yield eval(l)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        # device = self.get_device(batch)

        # step through environment with agent
        # reward, done = self.agent.play_step(self.net, epsilon, device)
        # self.episode_reward += reward
        # Instead of stepping through the environment, update the replay buffer with new games if possible
        self.average_value = self.update_replay_buffer()

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            print("Synced networks...")
            self.target_generation_net.load_state_dict(self.generation_net.state_dict())

        log = {'average_value': self.average_value,
               'steps': self.global_step,
               'buffer_size': len(self.buffer)}

        self.log("loss", loss, prog_bar=True)
        self.log("steps", self.global_step, prog_bar=True)
        self.log("buffer_size", len(self.buffer), prog_bar=True)
        self.log("average_value", self.average_value, prog_bar=True)

        time.sleep(1)

        return loss


    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        optimizer = optim.Adam(self.generation_net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=None,
            num_workers=5
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    # def on_epoch_end(self):
    #     trainer.save_checkpoint(os.path.join(args.root_dir, "checkpoints", "model.ckpt"))

class CustomCheckpointCallback(Callback):

    def on_batch_end(self, trainer, pl_module):
        if pl_module.global_step % pl_module.hparams.save_freq == 0:
            print("Saving checkpoint...")
            trainer.save_checkpoint(os.path.join(pl_module.hparams.root_dir, "checkpoints", "model.ckpt"))


def main(args) -> None:
    checkpoint_path = os.path.join(args.root_dir,"checkpoints","model.ckpt")
    model = DQNLightning.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.average_value = model.update_replay_buffer()

    trainer = Trainer(
        gpus=0,
        val_check_interval=10,
        default_root_dir = args.root_dir,
        checkpoint_callback = False,
        resume_from_checkpoint = checkpoint_path,
        callbacks=[CustomCheckpointCallback()]
    )

    trainer.fit(model)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str,
                        default='/home/jamison/projects/game_generation/results/my_model')

    args = parser.parse_args()

    main(args)