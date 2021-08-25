def inference(text,model,tokenizer,max=64,mask='[MASK]'):
    input  = tokenizer(text,max_length=max,padding='max_length',return_tensors='pt')['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input.numpy().squeeze())
    idx = tokens.index(mask)

    output = model(input)['last_hidden_state'].squeeze()
    masked_input = output[idx]
    predicted_vocab = np.argmax(masked_input)
    predicted_vocab = tokenizer.convert_ids_to_tokens([predicted_vocab])
    return predicted_vocab
