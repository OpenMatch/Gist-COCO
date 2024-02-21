from dataclasses import dataclass

from transformers import DataCollatorWithPadding, DefaultDataCollator
import torch
@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128
    max_sl_len: int = 2
    max_ll_len: int = 32

    def __call__(self, features):

        compression_input = [f["compression_input"] for f in features]
        label = [f["label"] for f in features]
        data_type = [f["data_type"] for f in features]


        if isinstance(compression_input[0], list):
            compression_input= sum(compression_input, [])
        if isinstance(label[0], list):
            label = sum(label[0],[])

        return compression_input, label,data_type