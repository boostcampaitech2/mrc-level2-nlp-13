##################
# import modules #
##################

from transformers import RobertaModel
from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.roberta.modeling_roberta import (
    ROBERTA_INPUTS_DOCSTRING,
    RobertaPreTrainedModel,
)
import torch.nn as nn
from torch.nn import CrossEntropyLoss

######################
# Classes & Funtions #
######################


class CnnHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels=input_size, out_channels=2, kernel_size=1, padding=0
        )
        self.conv_3 = nn.Conv1d(
            in_channels=input_size, out_channels=2, kernel_size=3, padding=1
        )
        self.conv_5 = nn.Conv1d(
            in_channels=input_size, out_channels=2, kernel_size=5, padding=2
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        x = conv1_out + conv3_out + conv5_out

        return x


class RobertaCNNForQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = CnnHead(input_size=config.hidden_size)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=QuestionAnsweringModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
