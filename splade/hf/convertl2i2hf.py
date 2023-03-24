from .args import ModelArguments, DataTrainingArguments, LocalTrainingArguments
import os

def convert(exp_dict):
    """
    Map  the l2i yaml  config into HF ModelArguments DataTrainingArguments  TrainingArguments
    input: Dictdefaut
    output: (ModelArguments, DataTrainingArguments,  TrainingArguments)


    config {'lr': 2e-05, 'seed': 123, 'gradient_accumulation_steps': 1, 'weight_decay': 0.01, 'validation_metrics': ['MRR@10', 'recall@100', 'recall@200', 'recall@500'], 'pretrained_no_yamlconfig': False, 'matching_type': 'splade', 'nb_iterations': 10, 'train_batch_size': 6, 'eval_batch_size': 8, 'index_retrieve_batch_size': 6, 'record_frequency': 3, 'train_monitoring_freq': 2, 'warmup_steps': 5, 'max_length': 10, 'fp16': False, 'loss': 'InBatchPairwiseNLL', 'augment_pairs': 'in_batch_negatives', 'monitoring_ckpt': {'metric': 'loss', 'type': 'FULL'}, 'regularizer': {'FLOPS': {'lambda_q': 0.0005, 'lambda_d': 0.0003, 'T': 3, 'targeted_rep': 'rep', 'reg': 'FLOPS'}}, 'tokenizer_type': 'distilbert-base-uncased', 'top_k': 5, 'threshold': 0.4, 'eval_metric': [['mrr_10', 'recall']], 'retrieval_name': ['toy'], 'checkpoint_dir': '???', 'index_dir': '???', 'out_dir': '???', 'hf_training': True}
    data {'type': 'triplets', 'TRAIN_DATA_DIR': 'toy_data/training_pairs', 'VALIDATION_SIZE_FOR_LOSS': 20, 'VALIDATION_RERANKING': {'DATA_DIR': 'toy_data/dev_set_for_training', 'QREL_PATH': 'toy_data/dev_set_for_training/qrel.json'},
        'COLLECTION_PATH': 'toy_data/doc_collection', 'Q_COLLECTION_PATH': {'toy': 'toy_data/eval_queries'}, 'EVAL_QREL_PATH': ['toy_data/qrel/qrel.json']}
    init_dict  {'model_type_or_dir': 'distilbert-base-uncased', 'model_type_or_dir_q': None, 'freeze_d_model': 0, 'agg': 'max', 'fp16': True}

    hf {'training': {'output_dir': '???'}, 'model': {'dense': True, 'distil': True}, 'data': {'document_dir': 'qdqsdsd'}}


    """

    config = exp_dict["config"]
    init_dict = exp_dict["init_dict"]
    data = exp_dict['data']
    hf = exp_dict["hf"]

    m = ModelArguments(init_dict.model_type_or_dir)
    d = DataTrainingArguments()
    t = LocalTrainingArguments()


    ############ model ############
    for k in hf.model.keys():
        setattr(m,k, hf.model[k])

    #will overwrite default/hf.yaml values    
    m.max_length = config.max_length
    #m.shared_weights = 
    #m.splade_doc = 
    m.model_q = init_dict.model_type_or_dir_q
    if config.matching_type == "siamese":
        m.dense_pooling = init_dict.output
        m.dense = config.matching_type == "siamese"

    # ad tokenizer to model_args ?
    m.tokenizer_name_or_path = config.tokenizer_type
    #m.adapter_name = 
    #m.adapter_config = 
    #m.load_adapter = 



    ############ data ############
    """
    type: mega_dataset
    TRAIN:
        DATASET_PATH: /nfs/data/neuralsearch/msmarco/scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz
        D_COLLECTION_PATH: /nfs/data/neuralsearch/msmarco/documents
        Q_COLLECTION_PATH: /nfs/data/neuralsearch/msmarco/training_queries/
        QREL_PATH: /nfs/data/neuralsearch/msmarco/training_queries/qrels.json



    type: triplets
    TRAIN_DATA_DIR: /nfs/data/neuralsearch/msmarco/triplets/

    """

    for k in hf.data.keys():
        setattr(d,k, hf.data[k])
    #will overwrite default/hf.yaml values    
    #d.result_path
    
    if data.type == "hard_negatives":
        d.scores = data.TRAIN.DATASET_PATH
        d.document_dir = os.path.join(data.TRAIN.D_COLLECTION_PATH,'raw.tsv') 
        d.query_dir = os.path.join(data.TRAIN.Q_COLLECTION_PATH,'raw.tsv')
        d.qrels_path = data.TRAIN.QREL_PATH
    elif data.type == "triplets":
        # currently command line hf.data
        pass

    #d.negatives_path
    #d.n_negatives
    #d.margin
    #d.top_k


    ############  training ############
    for k in hf.training.keys():
        setattr(t,k, hf.training[k])
    #will overwrite default/hf.yaml values    
    t.learning_rate = config.lr
    t.output_dir =  config.checkpoint_dir  
    #t.logging_dir = 
    t.per_device_train_batch_size =config.train_batch_size
    t.seed=config.seed
    #t.logging_steps = 
    #t.mse_margin = 

    ## nb_iterations  ->  learning_rate   


    if not m.dense:
       # here what if not l1q
       t.l0d  = config.regularizer.FLOPS.lambda_d 
       try: t.l0q = config.regularizer.L1.lambda_q
       except: # omegaconf.errors.ConfigKeyError
           t.l0q = t.l0d

    #t.save_total_limit


    return m, d, t


    
