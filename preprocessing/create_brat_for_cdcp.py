import os
import shutil
from pathlib import Path
from pre_process_utils import CDCPArgumentationDoc


def create_brat_from_original_cdcp_data(input_dir, output_dir):

    doc_ids = [file.split('.')[0] for file in os.listdir(input_dir) if file.split(".")[-1] == "txt"]

    file_count = 0
    total_rel_count = 0
    total_comp_count = 0
    for doc_id in doc_ids:

        doc = CDCPArgumentationDoc(input_dir + '/' + doc_id)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_dir + '/' + doc_id + '.ann', 'w') as writer:

            component_count = 0
            relation_count = 0

            for comp_idx, (begin, end) in enumerate(doc.prop_offsets):
                component_count += 1
                writer.write("T" + str(comp_idx + 1) + "\t" + doc.prop_labels[comp_idx] + " " + str(begin) + " " + str(
                    end) + "\t" + doc.raw_text[begin:end] + "\n")

            for src_comp, trg_comp in doc.evidences:
                relation_count += 1
                writer.write("R" + str(relation_count) + "\t" + "evidence" + " " + "Arg1:T" + str(
                    src_comp + 1) + " " + "Arg2:T" + str(trg_comp + 1) + "\t" + "\n")

            for src_comp, trg_comp in doc.reasons:
                relation_count += 1
                writer.write("R" + str(relation_count) + "\t" + "reason" + " " + "Arg1:T" + str(
                    src_comp + 1) + " " + "Arg2:T" + str(trg_comp + 1) + "\t" + "\n")

            total_rel_count += relation_count
            total_comp_count += component_count

        assert component_count == len(doc.prop_labels)
        assert relation_count == len(doc.reasons) + len(doc.evidences)
        shutil.copyfile(input_dir + '/' + doc_id + '.txt', output_dir + '/' + doc_id + '.txt')
        file_count += 1

    print("{} docs output to {}".format(file_count, output_dir))
    print("Contains {} relations and {} components.".format(total_rel_count, total_comp_count))


def main():
    create_brat_from_original_cdcp_data('../data/cdcp/original/dev', '../data/cdcp/brat/dev')
    create_brat_from_original_cdcp_data('../data/cdcp/original/train', '../data/cdcp/brat/train')
    create_brat_from_original_cdcp_data('../data/cdcp/original/test', '../data/cdcp/brat/test')


if __name__ == "__main__":
    main()
