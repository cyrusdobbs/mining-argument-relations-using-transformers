from hyperopt import hp, pyll, fmin, Trials, tpe

import train


def main():


    # MODEL_CLASSES = {
    #     "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    #     "bert-rbert": (BertConfig, RBERT, BertTokenizer),
    #     "bert-distance": (BertConfig, BertForSequenceClassificationWithDistance, BertTokenizer),
    #     "bert-distance-components": (
    #     BertConfig, BertForSequenceClassificationWithDistanceAndComponentType, BertTokenizer),
    #     "bert-distance-components-new": (
    #     BertConfig, BertForSequenceClassificationWithDistanceAndComponentTypeNew, BertTokenizer),
    #     "bert-resnet": (BertConfig, BertForSequenceClassificationResNet, BertTokenizer),
    #     "bert-resnet-jl": (BertConfig, BertForSequenceClassificationResNetJointLearning, BertTokenizer),
    #     "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    #     "bert-seqtag": (BertConfig, BertForSequenceTagging, ExtendedBertTokenizer),
    # }
    # "cdcp_seqtag": ArgMinSeqTagProcessorCDCP,
    # "seqtag": ArgMinSeqTagProcessor,
    # "relclass": ArgMinRelClassProcessor,
    # "cdcp_relclass": ArgMinRelClassProcessorCDCP,
    # "multichoice": (ArgMinRelClassForMultiChoiceProcessor, ArgMinMultiChoiceLinkProcessor),
    # "cdcp_relclass_context": ArgMinRelClassProcessorCDCPWithContext,
    # "cdcp_relclass_rbert": ArgMinRelClassProcessorCDCPRBERT,
    # "cdcp_relclass_distance": ArgMinRelClassProcessorCDCPWithDistance,
    # "cdcp_relclass_distance_components": ArgMinRelClassProcessorCDCPWithDistanceAndComponentType,
    # "cdcp_relclass_resnet": ArgMinRelClassProcessorCDCPResNet,
    # "cdcp_relclass_resnet_jl": ArgMinRelClassProcessorCDCPResNetJointLearning,
    # "cdcp_relclass_resnet_jl_inv": ArgMinRelClassProcessorCDCPResNetJointLearningInverseLabels

    space = {'model_specific': hp.choice('model_type', [
        {
            'task_name': 'cdcp_relclass',
            'model_type': 'bert',
        },
        # {
        #     'task_name': 'cdcp_relclass_context',
        #     'model_type': 'bert',
        # },
        # {
        #     'task_name': 'cdcp_relclass_rbert',
        #     'model_type': 'bert-rbert',
        # },
        {
            'task_name': 'cdcp_relclass_distance',
            'model_type': 'bert-distance',
            'classifier_type': hp.choice('dist_classifier', ['FC', '2FC', '2FC-RelU', '2FC-Tanh'])
        },
        {
            'task_name': 'cdcp_relclass_distance_components',
            'model_type': 'bert-distance-components-new',
            'classifier_type': hp.choice('dist_comp_classifier', ['FC', '2FC', '2FC-RelU', '2FC-Tanh'])
        },
        # {
        #     'task_name': 'cdcp_relclass_resnet',
        #     'model_type': 'bert-resnet',
        # },
        # {
        #     'task_name': 'cdcp_relclass_resnet_jl',
        #     'model_type': 'bert-resnet-jl',
        # },
        # {
        #     'task_name': 'cdcp_relclass_resnet_jl_inv',
        #     'model_type': 'bert-resnet-jl',
        # },
    ]),
             'max_seq_length': hp.choice('seq_len', [128, 256, 166]),
             'learning_rate': hp.choice('lr', [5e-5, 4e-5, 3e-5, 2e-5]),
             'freeze_bert': hp.choice('freeze', [
                 {'not_frz': False, 'epochs': hp.choice('not_frz_ep', [2, 3, 4])},
                 {'frz': True, 'epochs': hp.choice('ep', [3, 5, 10, 20, 30])}
             ]),
             'weight_decay': hp.choice('wd', [0.00001, 0.0001, 0.001, 0.01])
    }

    def run_training(space):
        train.main(space)

    this_space = pyll.stochastic.sample(space)
    print(this_space)
    train.main(this_space)
    # trials = Trials()
    # best = fmin(run_training,
    #             space=space,
    #             algo=tpe.suggest,
    #             max_evals=100,
    #             trials=trials)
    #
    # print(best)

if __name__ == '__main__':
    main()