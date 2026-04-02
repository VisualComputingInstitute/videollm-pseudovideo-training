from __future__ import annotations
from abc import ABC, abstractmethod
import random
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any, Type, TypeVar, Optional, Callable

T = TypeVar('T', bound='QADataset')


@dataclass
class Question:
    corresponding_data: Any
    question: str
    answer: str
    candidate_answers: List[str] = None


class QuestionGenerator(ABC):

    @abstractmethod
    def setup_on_dataset(self, dataset: QADataset):
        pass

    @abstractmethod
    def form_question(self, question_index: int) -> Question:
        pass

    @abstractmethod
    def get_length(self) -> int:
        pass

class MCAndFreeFormQGen(QuestionGenerator):

    def __init__(self, mc_question_prob: float=1):
        if(mc_question_prob < 0 or mc_question_prob > 1):
            raise ValueError("`mc_question_prob` must be in [0, 1]")
        self.mc_question_prob = mc_question_prob

    @abstractmethod
    def form_mc_question(self, question_index: int) -> Question:
        pass

    @abstractmethod
    def form_free_form_question(self, question_index: int) -> Question:
        pass

    def form_question(self, question_index: int) -> Question:
        assert question_index >= 0
        rand = random.uniform(0, 1)
        if(rand < self.mc_question_prob):
            return self.form_mc_question(question_index)
        else:
            return self.form_free_form_question(question_index)

class QADataset(Dataset):
    known_question_generators: Dict[str, Callable[..., QuestionGenerator]] = {}
    question_generators: List[QuestionGenerator]

    def __init__(self, also_return_answers: bool=False):
        self.total_length = None
        self.sampling_prob_per_sample = None
        self.also_return_answers = also_return_answers  

    @classmethod
    def register_question_generator(
        cls: Type[QADataset],
        q_gen_name: str, 
        q_gen_creator: Callable[..., QuestionGenerator]):

        if(q_gen_name in cls.known_question_generators):
            raise Exception("\"{0}\" is already mapped to a question "
                            "generator".format(q_gen_name))
        cls.known_question_generators[q_gen_name] = q_gen_creator


    def get_sampling_prob_per_sample(self) -> Optional[List[float]]:
        return self.sampling_prob_per_sample        

    def __getitem__(
        self, i: int) -> Tuple[
            Any, str, str, Optional[List[str]]]:
        """Returns a tuple consisting of information about the data
        sample (type is "Any" since this depends on the subclass), the 
        question, the answer, and a list of all candidate answers that 
        are part of the question text if `also_return_answers` was 
        `True` in `__init__`, in which case it is assumed that the 
        question is in multiple-choice format.
        """
        if(self.total_length is None):
            raise ValueError("The dataset has not been equipped with any "
                             "question generators")        
        if(i < self.total_length):
            gen_ind = next(
                filter(lambda _: self.cumulative_lens[_] > i, 
                       range(len(self.cumulative_lens)))) - 1
            q = self.question_generators[gen_ind].form_question(
                i - self.cumulative_lens[gen_ind])
            return ((q.corresponding_data, q.question, q.answer) + 
                    ((q.candidate_answers,) if self.also_return_answers 
                     else ()))
        else:
            raise IndexError

    def __len__(self):
        if(self.total_length is None):
            raise ValueError("The dataset has not been equipped with any "
                             "question generators")
        return self.total_length    

    def setup_with_question_generators(
            self, 
            question_generators: List[QuestionGenerator],
            sampling_prob_per_gen: Optional[List[float]]=None):
        if(sampling_prob_per_gen is not None):
            if(any(sampling_prob_per_gen) < 0 or any(sampling_prob_per_gen) > 1
               or abs(sum(sampling_prob_per_gen) - 1) > 1e-5):
                raise ValueError("`sampling_prob_per_gen` should be a valid "
                                "probability distribution")
            if(len(sampling_prob_per_gen) != len(question_generators)):
                raise ValueError(
                    "The number of sampling probabilities should be equal to "
                    "the number of question generators")
        
        self.question_generators = question_generators
        self.sampling_prob_per_gen = sampling_prob_per_gen
        for q_gen in self.question_generators:
            q_gen.setup_on_dataset(self)
        self.qgen_lengths = [_.get_length() for _ in question_generators]
        self.cumulative_lens = [0]
        for _ in self.qgen_lengths:
            self.cumulative_lens.append(_ + self.cumulative_lens[-1])
            
        self.total_length = sum(self.qgen_lengths)
        assert (self.cumulative_lens[-1] == self.total_length
                and len(self.cumulative_lens ) == len(self.qgen_lengths) + 1) 
        if(self.sampling_prob_per_gen is not None):
            gen_ids = self._get_generator_id_per_sample()
            self.sampling_prob_per_sample = [
                self.sampling_prob_per_gen[_]/self.qgen_lengths[_] 
                for _ in gen_ids]
            assert (len(self.sampling_prob_per_sample) == self.total_length
                    and abs(sum(self.sampling_prob_per_sample) - 1) < 1e-5) 
               
    @classmethod
    def from_config_KVs(
        cls: Type[QADataset], 
        q_generators_args: Dict[str, Dict[str, Any]],
        sampling_prob_per_generator: Optional[Dict[str, float]]=None,
        **constructor_kwargs) -> QADataset:
        unpacked = [
            (gen, gen_args) for gen, gen_args in q_generators_args.items()]
        q_gens = [cls.known_question_generators[gen](**gen_args)
                  for gen, gen_args in unpacked]
        if(sampling_prob_per_generator is not None):
            gen_probs = [
                sampling_prob_per_generator[gen] for gen, _ in unpacked]
        else:
            gen_probs = None
        obj = cls(**constructor_kwargs)
        obj.setup_with_question_generators(q_gens, gen_probs)

        return obj

    def _get_generator_id_per_sample(self):
        if(self.total_length is None):
            raise ValueError("The dataset has not been equipped with any "
                             "question generators")
        ret = [id for id, qgen_len in enumerate(self.qgen_lengths) 
               for _ in range(qgen_len)]
        assert len(ret) == self.total_length
        return ret