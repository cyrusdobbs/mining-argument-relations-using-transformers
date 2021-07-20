import os
import nltk
from pathlib import Path
from pre_process_utils import CDCPArgumentationDoc


def create_connl_from_original_cdcp_data(input_dir, output_file, short_test = False):
    doc_ids = [file.split('.')[0] for file in os.listdir(input_dir) if file.split(".")[-1] == "txt"]

    docs = []
    total_rel_count = 0
    total_comp_count = 0
    for doc_id in doc_ids:

        doc = CDCPArgumentationDoc(input_dir + '/' + doc_id)

        assert len(doc.prop_offsets) == len(doc.prop_labels)

        total_rel_count += len(doc.links)
        total_comp_count += len(doc.prop_labels)

        components = []
        for offset_idx, (begin, end) in enumerate(doc.prop_offsets):

            if begin == 0:
                components.append((doc.raw_text[begin:end], doc.prop_labels[offset_idx]))

            elif offset_idx == 0:
                components.append((doc.raw_text[0:begin], None))
                components.append((doc.raw_text[begin:end], doc.prop_labels[offset_idx]))

            elif begin == doc.prop_offsets[offset_idx - 1][1]:
                components.append((doc.raw_text[begin:end], doc.prop_labels[offset_idx]))

            else:
                components.append((doc.raw_text[doc.prop_offsets[offset_idx - 1][1]:begin], None))
                components.append((doc.raw_text[begin:end], doc.prop_labels[offset_idx]))

        if len(doc.raw_text) > doc.prop_offsets[-1][1]:
            components.append((doc.raw_text[doc.prop_offsets[-1][1]:len(doc.raw_text)], None))

        if len(doc.raw_text) != len(''.join([component[0] for component in components])):
            raise Exception('Document components do not make up whole doc.')

        docs.append(components)

    tags = {'policy': ['B-Policy', 'I-Policy'],
            'value': ['B-Value', 'I-Value'],
            'testimony': ['B-Testimony', 'I-Testimony'],
            'fact': ['B-Fact', 'I-Fact'],
            'reference': ['B-Reference', 'I-Reference']}

    Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as writer:
        doc_count = 0
        for doc in docs:
            if short_test and doc_count > 5:
                break
            token_count = 0
            for component, label in doc:
                tokenized = nltk.word_tokenize(component)
                first_token = True
                for token in tokenized:
                    if label is None:
                        writer.write(str(token_count) + '\t' + token + '\t' + '_' + '\t' + '_' '\t' + 'O' + '\n')
                    elif first_token:
                        writer.write(str(token_count) + '\t' + token + '\t' + '_' + '\t' + '_' + '\t' + str(tags[label][0]) + '\n')
                        first_token = False
                    else:
                        writer.write(str(token_count) + '\t' + token + '\t' + '_' + '\t' + '_' + '\t' + str(tags[label][1]) + '\n')
                    token_count += 1
                writer.write('\n')
            writer.write('\n')
            doc_count += 1

    print("{} docs output to {}".format(doc_count, output_file))
    print("Contains {} relations and {} components.".format(total_rel_count, total_comp_count))





def main():
    short_test = False

    if not short_test:
        create_connl_from_original_cdcp_data('../data/cdcp/original/dev', '../data/cdcp/sequence_tags/dev.conll')
        create_connl_from_original_cdcp_data('../data/cdcp/original/train', '../data/cdcp/sequence_tags/train.conll')
        create_connl_from_original_cdcp_data('../data/cdcp/original/test', '../data/cdcp/sequence_tags/test.conll')
    else:
        create_connl_from_original_cdcp_data('../data/cdcp/original/dev', '../data/cdcp/sequence_tags_short_test/dev.conll', short_test)
        create_connl_from_original_cdcp_data('../data/cdcp/original/train', '../data/cdcp/sequence_tags_short_test/train.conll', short_test)
        create_connl_from_original_cdcp_data('../data/cdcp/original/test', '../data/cdcp/sequence_tags_short_test/test.conll', short_test)


if __name__ == "__main__":
    main()
