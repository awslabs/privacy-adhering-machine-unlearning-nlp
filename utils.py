import torch

def tokenize_and_get_tesnors(tokenizer, input, device, use_roberta = False):
    if len(input[0]) == 2:
        features = tokenizer([x[0] for x in input], [x[1] for x in input] , max_length=256, truncation=True, padding=True)
    else:
        features = tokenizer([x[0] for x in input], max_length=256, truncation=True,
                             padding=True)

    all_input_ids = torch.tensor([f for f in features["input_ids"]], dtype=torch.long)
    all_attention_mask = torch.tensor([f for f in features["attention_mask"]], dtype=torch.long)
    all_input_ids = all_input_ids.to(device)
    all_attention_mask = all_attention_mask.to(device)

    if not use_roberta:
        all_token_type_ids = torch.tensor([f for f in features["token_type_ids"]], dtype=torch.long)
        all_token_type_ids = all_token_type_ids.to(device)
    else:
        all_token_type_ids = None

    if not use_roberta:
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "token_type_ids": all_token_type_ids}
    else:
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_mask}

    return inputs
