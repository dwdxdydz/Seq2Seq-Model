def translate_sentence(sentence, model, src_vocab, trg_vocab, device, max_length=50):
    
    # Preprocess the sentence
    tokens = tokenize(sentence)
    src_ids = [src_vocab[token] for token in tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = torch.ones(1, src_tensor.size(1), dtype=torch.long, device=device)

    # Initialize decoder input (BOS token)
    trg_index = trg_vocab['<bos>']
    trg_tensor = torch.tensor([trg_index], dtype=torch.long, device=device).unsqueeze(0)

    # Translate with greedy search
    with torch.no_grad():
        for t in range(max_length):
            output, hidden, _ = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)
            pred = output.argmax(1)
            trg_index = pred.item()

            # If end of sentence (EOS) token is predicted, stop
            if trg_index == trg_vocab['<eos>']:
                break

            # Update target tensor for next prediction
            trg_tensor = torch.cat((trg_tensor, torch.tensor([trg_index], device=device).unsqueeze(0)), dim=1)

        # Convert translated IDs to words
        translated_sentence = [trg_vocab.itos[i] for i in trg_tensor.squeeze().tolist()]

        # Remove padding tokens
        if '<pad>' in translated_sentence:
            translated_sentence = translated_sentence[:translated_sentence.index('<pad>')]

    return ' '.join(translated_sentence)