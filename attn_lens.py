def tokenize_series(series, tokenizer):
    input_ids, attention_mask, _ = tokenizer.context_input_transform(series)
    return input_ids, attention_mask


def attn_lens(input_ids, attention_mask, t5_model, max_new_tokens=10, decoder_input_ids=None):
    # dict to store the cross attention probabilities
    # key is the pass number, value is a list of cross attention probabilities
    cross_attn_probs = {}

    # Hook to save the cross attention probabilities
    def save_cross_probs(module, inputs, outputs):
        # T5Attention.forward returns a tuple:
        #   (attn_output, present_key_value, position_bias, attn_probs)
        #   outputs[3] is the attention probabilities (batch, heads, tgt_len, src_len)
        pass_num = outputs[3].shape[2] - 1
        cross_attn_probs[pass_num] = cross_attn_probs.get(pass_num, []) + [outputs[3].detach().cpu()]

    hooks = []
    for block in t5_model.decoder.block:
        encdec_attn = block.layer[1].EncDecAttention
        hooks.append(encdec_attn.register_forward_hook(save_cross_probs))


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
    
    for h in hooks:
        h.remove()

    return outputs, cross_attn_probs

