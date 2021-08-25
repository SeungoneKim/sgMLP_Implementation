def inference(text,model,tokenizer,max=64,mask='[MASK]'):
    input  = tokenizer(text,max_length=max,padding='max_length',return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(input['input_ids'].numpy().squeeze())
    idx = tokens.index(mask)
    output = model(input['input_ids'],input['token_type_ids']).squeeze()
    masked_input = output[idx].detach().numpy()
    predicted_vocab = np.argmax(masked_input)
    predicted_vocab = tokenizer.convert_ids_to_tokens([predicted_vocab])
    return predicted_vocab
  
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = build_model(tokenizer.vocab_size,512,2048,64,12,'cpu')
weight = torch.load('iter_40000.pth',map_location=torch.device('cpu'))['model_state_dict']
model_weight = {}
for key,val in weight.items():
    if key.startswith('module.'):
        model_weight[key[7:]] = val
    else:
        print(key)
model.load_state_dict(model_weight)
