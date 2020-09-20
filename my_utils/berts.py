from transformers import *
from torch import nn

def RoBERTa_Large_gets():
    pretrained_weights = "roberta-large"
    config = RobertaConfig.from_pretrained(pretrained_weights)
    config.output_hidden_states = True
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
    base_model = RobertaForSequenceClassification.from_pretrained(pretrained_weights, config=config)
    base_model.classifier.out_proj = torch.nn.Linear(1024, 4)
    nn.init.normal_(base_model.classifier.out_proj.weight, 0.0, 0.02)
    nn.init.constant_(base_model.classifier.out_proj.bias, 0)

    # base_model.resize_token_embeddings(new_num_tokens=50265)
    return base_model, tokenizer, {"isbert": False, "albert": False, "isxlnet": False, "isJoBERTa": False}


def BART_gets():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = AutoModelWithLMHead.from_pretrained("facebook/bart-base")
    model.config.activation_dropout = 0.1
    model.config.attention_dropout = 0.1
    base_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base", config=model.config)
    base_model.classification_head.out_proj = torch.nn.Linear(768, 4)
    nn.init.normal_(base_model.classification_head.out_proj.weight, 0.0, 0.02)
    nn.init.constant_(base_model.classification_head.out_proj.bias, 0)

    return base_model, tokenizer, {"isbert": False, "albert": False, "isxlnet": False, "isJoBERTa": False}


def BERT_gets():
    pretrained_weights = "bert-base-uncased"
    config = BertConfig.from_pretrained(pretrained_weights)
    config.output_hidden_states = True
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    base_model = BertForSequenceClassification.from_pretrained(pretrained_weights, config=config)
    base_model.classifier = torch.nn.Linear(768, 4)
    nn.init.normal_(base_model.classifier.weight, 0.0, 0.02)
    nn.init.constant_(base_model.classifier.bias, 0)
    return base_model, tokenizer, {"isbert": True, "albert": False, "isxlnet": False, "isJoBERTa": False}


def XLnet_gets():
    pretrained_weights = "xlnet-base-cased"
    config = XLNetConfig.from_pretrained(pretrained_weights)
    config.output_hidden_states = True
    tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
    base_model = XLNetForSequenceClassification.from_pretrained(pretrained_weights, config=config)
    base_model.logits_proj = torch.nn.Linear(768, 4)
    nn.init.normal_(base_model.logits_proj.weight, 0.0, 0.02)
    nn.init.constant_(base_model.logits_proj.bias, 0)

    return base_model, tokenizer, {"isbert": False, "albert": False, "isxlnet": True, "isJoBERTa": False}


def ALBERT_gets():
    pretrained_weights = "albert-large-v2"
    config = AlbertConfig.from_pretrained(pretrained_weights)
    config.output_hidden_states = True
    tokenizer = AlbertTokenizer.from_pretrained(pretrained_weights)
    base_model = AlbertForSequenceClassification.from_pretrained(pretrained_weights, config=config)
    base_model.classifier = torch.nn.Linear(1024, 4)
    nn.init.normal_(base_model.classifier.weight, 0.0, 0.02)
    nn.init.constant_(base_model.classifier.bias, 0)

    return base_model, tokenizer, {"isbert": False, "albert": True, "isxlnet": False, "isJoBERTa": False}


def ELECTRA_gets():
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    base_model = AutoModelForSequenceClassification.from_pretrained("google/electra-base-discriminator")
    base_model.classifier.out_proj = torch.nn.Linear(768, 4)
    nn.init.normal_(base_model.classifier.out_proj.weight, 0.0, 0.02)
    nn.init.constant_(base_model.classifier.out_proj.bias, 0)
    return base_model, tokenizer, {"isbert": True, "albert": False, "isxlnet": False, "isJoBERTa": False}


def JoBERTa_gets():
    base_model = AutoModelForSequenceClassification.from_pretrained("MoseliMotsoehli/JoBerta", from_tf=True)
    base_model.config.output_hidden_states = True
    base_model = AutoModelForSequenceClassification.from_pretrained("MoseliMotsoehli/JoBerta", config=base_model.config,
                                                                    from_tf=True)
    tokenizer = RobertaTokenizer("./vocab.json", "merges.txt")
    base_model.classifier.out_proj = torch.nn.Linear(768, 4)
    nn.init.normal_(base_model.classifier.out_proj.weight, 0.0, 0.02)
    nn.init.constant_(base_model.classifier.out_proj.bias, 0)

    return base_model, tokenizer, {"isbert": False, "albert": False, "isxlnet": False, "isJoBERTa": True}

def finetune_optimizer_state(model,freeze_layer_num = 0,weight_decay=0.001):
    """
    :param model: transformers_model
    :param freeze_layer_num:int
    :return:optimizer_state

    example:
    optimizer_state = finetune_optimizer_state(model,freeze_layer_num=0)
    optimizer = torch.optim.AdamW(optimizer_state, lr=2e-5 weight_decay=0.001, eps=1e-8)
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    if freeze_layer_num>0:
        freeze_layer = [f"encoder.layer.{i}." for i in range(freeze_layer_num)]
    else:
        freeze_layer = ["z"]

    for n, p in model.bert.named_parameters():
        if any([nd in n for nd in freeze_layer]):
            p.requires_grad = False
            # print(n)

    bert_params1 = {
        "params": [p for n, p in list(model.named_parameters()) if
                   not any(nd in n for nd in no_decay) and not any(nd in n for nd in freeze_layer)],
        "weight_decay": weight_decay,
    }
    bert_params2 = {
        "params": [p for n, p in list(model.named_parameters()) if
                   any(nd in n for nd in no_decay) and not any(nd in n for nd in freeze_layer)],
        "weight_decay": 0.0
    }
    optimizer_state = [bert_params1, bert_params2]
    # optimizer = torch.optim.AdamW(optimizer_state, lr=base_lr, weight_decay=0.001, eps=1e-8)
    return optimizer_state