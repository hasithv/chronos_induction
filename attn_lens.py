def tokenize_series(series, tokenizer):
    input_ids, attention_mask, _ = tokenizer.context_input_transform(series)
    return input_ids, attention_mask


def attn_lens(input_ids, attention_mask, t5_model, max_new_tokens=10, decoder_input_ids=None):
    # dict to store the cross attention probabilities
    # key is the pass number, value is a list of cross attention probabilities
    cross_attn_probs = {}

    outputs = t5_model.generate(input_ids=input_ids, 
                      attention_mask=attention_mask, 
                      max_new_tokens=max_new_tokens,
                      decoder_input_ids=decoder_input_ids,
                      num_return_sequences=1,
                      do_sample=False,
                      use_cache=False,
                      output_attentions=True,
                      output_scores=True,
                      output_hidden_states=True,
                      return_dict_in_generate=True
                      )
    
    for i in range(len(outputs.cross_attentions)):
        # the k-th key represents the pass number
        # for example, if k=1, then it means we are looking at the cross attentions when the decoder input has
        # an input of [<dec_start>, t_1, t_2, ..., t_k]
        k = i + decoder_input_ids.shape[-1] - 1 if decoder_input_ids is not None else i
        cross_attn_probs[k] = [outputs.cross_attentions[i][j].detach().cpu() for j in range(len(outputs.cross_attentions[i]))]

    return outputs, cross_attn_probs

