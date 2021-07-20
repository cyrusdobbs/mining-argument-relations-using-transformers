from pre_process_utils import CDCPArgumentationDoc
import os


def get_longest_sequence(input_dir):

    doc_ids = [file.split('.')[0] for file in os.listdir(input_dir) if file.split(".")[-1] == "txt"]

    largest_offset = 0
    average_offset = 0
    count = 0
    largest_offset_doc = None
    for doc_id in doc_ids:

        doc = CDCPArgumentationDoc(input_dir + '/' + doc_id)

        for offsets in doc.prop_offsets:
            current_offset = offsets[1] - offsets[0]
            if count == 0:
                average_offset = current_offset
            else:
                average_offset = (average_offset * count + current_offset) / (count + 1)

            if offsets[1] - offsets[0] > largest_offset:
                largest_offset = offsets[1] - offsets[0]
                largest_offset_doc = doc

            count += 1

    print("Longest sequence is {} characters.".format(largest_offset))
    print("Average sequence is {} characters.".format(average_offset))
    print(largest_offset_doc.raw_text)
    print(largest_offset_doc.prop_offsets)
    print(largest_offset_doc.reasons)
    print(largest_offset_doc.evidences)
    print(largest_offset_doc.doc_id)
    print(len(largest_offset_doc.raw_text))

def main():

    get_longest_sequence('../data/cdcp/original/dev')
    get_longest_sequence('../data/cdcp/original/train')
    get_longest_sequence('../data/cdcp/original/test')


if __name__ == "__main__":
    main()