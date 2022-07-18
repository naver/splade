from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def init_simple_bert_optim(model, lr, weight_decay, warmup_steps, num_training_steps):
    """
    inspired from https://github.com/ArthurCamara/bert-axioms/blob/master/scripts/bert.py
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler
