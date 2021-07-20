import os
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import nltk
import csv
import random
from pathlib import Path

def train_test_split(data_dir):
    """Function thats splits a folder with brat files into train/test based on csv file"""

    csv_split = 'train-test-split.csv'

    with open(data_dir+csv_split, newline='') as f:
        reader = csv.reader(f, delimiter=';')

        for row in reader:
            if row[1] == 'TRAIN':
                if os.path.isfile(data_dir+row[0]+'.txt') and os.path.isfile(data_dir+row[0]+'.ann'):
                    os.rename(data_dir + row[0] + '.txt', data_dir + 'train/' + row[0] + '.txt')
                    os.rename(data_dir + row[0] + '.ann', data_dir + 'train/' + row[0] + '.ann')
            elif row[1] == 'TEST':
                if os.path.isfile(data_dir + row[0]+'.txt') and os.path.isfile(data_dir+row[0]+'.ann'):
                    os.rename(data_dir + row[0] + '.txt', data_dir + 'test/' + row[0] + '.txt')
                    os.rename(data_dir + row[0] + '.ann', data_dir + 'test/' + row[0] + '.ann')

def create_tsv_for_relation_classification(input_dir, output_file):
    """
    Function to create a tsv file with random negative samples from corpus.
    :param input_dir: directory with .txt and .ann files
    :param output_file: tsv file to write the output to
    :return: None
    """

    total_rel_count = 0
    total_comp_count = 0
    for f in os.listdir(input_dir):
        if '.ann' in f:
            links = {}
            relations = {}
            components = {}
            pairs = []
            f_name = f[:-4]

            # open ann file and get components and relations
            with open(input_dir + f_name + '.ann', 'r', encoding='utf-8') as fann:
                lines = fann.read().splitlines()
                for line in lines:
                    line = line.split('\t')
                    if 'T' in line[0]:
                        components[line[0]] = line[2].replace('\n', '')
                    elif 'R' in line[0]:
                        rel = line[1].split()[1:]
                        links[rel[0][5:]] = rel[1][5:]
                        relations[rel[0][5:]] = line[1].split()[0]

            assert len(relations.keys()) == len(links.keys())

            for c_i in components.keys():
                for c_j in components.keys():
                    if c_i == c_j:
                        continue
                    elif c_i in links.keys() and c_j == links[c_i]:
                        pairs.append('__label__'+relations[c_i] + '\t' + components[c_i] + '\t' + components[c_j])
                    else:
                        pairs.append('__label__noRel\t' + components[c_i] + '\t' + components[c_j])

            assert len(components.keys()) * len(components.keys()) - len(components.keys()) == len(pairs)

            Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

            with open(output_file + '.tsv', 'a', encoding='utf-8') as fout:
                for pair in pairs:
                    fout.write(pair + '\n')

            total_rel_count += len(relations)
            total_comp_count += len(components)

    print("TSV output to {}".format(output_file + '.tsv'))
    print("Contains {} relations and {} components.".format(total_rel_count, total_comp_count))


def create_tsv_for_relation_classificationCDCP(input_dir, output_file, sample_for_testing=False):
    """
    Function to create a tsv file with random negative samples from corpus.
    :param input_dir: directory with .txt and .ann files
    :param output_file: tsv file to write the output to
    :return: None
    """

    total_rel_count = 0
    total_comp_count = 0
    for doc_count, f in enumerate(os.listdir(input_dir)):
        if sample_for_testing and doc_count > 20:
            break

        if '.ann' in f:
            links = {}
            relations = {}
            components = {}
            pairs = []
            f_name = f[:-4]

            # open ann file and get components and relations
            with open(input_dir + f_name + '.ann', 'r', encoding='utf-8') as fann:
                lines = fann.read().splitlines()
                for line in lines:
                    line = line.split('\t')
                    if 'T' in line[0]:
                        components[line[0]] = line[2].replace('\n', '')
                    elif 'R' in line[0]:
                        rel = line[1].split()[1:]
                        if rel[0][5:] in links:
                            links[rel[0][5:]].append(rel[1][5:])
                            relations[rel[0][5:]].append(line[1].split()[0])
                        else:
                            links[rel[0][5:]] = [rel[1][5:]]
                            relations[rel[0][5:]] = [line[1].split()[0]]

            assert len(relations.keys()) == len(links.keys())

            for c_i in components.keys():
                for c_j in components.keys():
                    if c_i == c_j:
                        continue
                    elif c_i in links.keys() and c_j in links[c_i]:
                        total_rel_count += 1
                        pairs.append('__label__'+relations[c_i][links[c_i].index(c_j)] + '\t' + components[c_i] + '\t' + components[c_j])
                    else:
                        pairs.append('__label__noRel\t' + components[c_i] + '\t' + components[c_j])

            assert len(components.keys()) * len(components.keys()) - len(components.keys()) == len(pairs)

            Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

            with open(output_file + '.tsv', 'a', encoding='utf-8') as fout:
                for pair in pairs:
                    fout.write(pair + '\n')

            total_comp_count += len(components)

    print("TSV output to {}".format(output_file + '.tsv'))
    print("Contains {} relations and {} components.".format(total_rel_count, total_comp_count))

def create_tsv_for_relation_classificationCDCPWithContext(input_dir, output_file):
    """
    Function to create a tsv file with random negative samples from corpus.
    :param input_dir: directory with .txt and .ann files
    :param output_file: tsv file to write the output to
    :return: None
    """

    total_rel_count = 0
    total_comp_count = 0
    for f in os.listdir(input_dir):
        if '.ann' in f:
            links = {}
            relations = {}
            components = {}
            pairs = []
            f_name = f[:-4]

            # open ann file and get components and relations
            with open(input_dir + f_name + '.ann', 'r', encoding='utf-8') as fann:
                lines = fann.read().splitlines()
                for line in lines:
                    line = line.split('\t')
                    if 'T' in line[0]:
                        components[line[0]] = line[2].replace('\n', '')
                    elif 'R' in line[0]:
                        rel = line[1].split()[1:]
                        if rel[0][5:] in links:
                            links[rel[0][5:]].append(rel[1][5:])
                            relations[rel[0][5:]].append(line[1].split()[0])
                        else:
                            links[rel[0][5:]] = [rel[1][5:]]
                            relations[rel[0][5:]] = [line[1].split()[0]]

            assert len(relations.keys()) == len(links.keys())

            for c_i in components.keys():
                for c_j in components.keys():
                    if c_i == c_j:
                        continue
                    elif c_i in links.keys() and c_j in links[c_i]:
                        total_rel_count += 1
                        pairs.append('__label__'+relations[c_i][links[c_i].index(c_j)] + '\t' + components[c_i] + '\t' + components[c_j])
                    else:
                        pairs.append('__label__noRel\t' + components[c_i] + '\t' + components[c_j])

            assert len(components.keys()) * len(components.keys()) - len(components.keys()) == len(pairs)

            Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

            with open(output_file + '.tsv', 'a', encoding='utf-8') as fout:
                for pair in pairs:
                    fout.write(pair + '\n')

            total_comp_count += len(components)

    print("TSV output to {}".format(output_file + '.tsv'))
    print("Contains {} relations and {} components.".format(total_rel_count, total_comp_count))

def create_tsv_for_multiple_choice(input_dir, output_file, num_of_choices=5):
    """
    Function to create a tsv file with random negative samples from corpus. For all component including unconnected ones.
    Last option is always the no link option.
    :param input_dir: directory with .txt and .ann files
    :param output_file: tsv file to write the output to
    :return: None
    """

    if os.path.isfile(output_file + '.tsv'):
        os.remove(output_file + '.tsv')

    def _build_choices(gold_pair, components):
        """
        creates answers for a multiple choice question where the wrong answers are in components
        :param gold_pair: tuple of subj with correct obj
        :param components: list of all possible components
        :return: list of possible answers with position of correct answer
        """

        subj, obj = gold_pair
        choices = []

        idx = 0
        while len(choices) < num_of_choices:
            if idx == len(components):
                idx = 0
            comp = components[idx]
            if comp == obj or comp == subj or comp == 'T0':
                idx += 1
                continue
            else:
                idx += 1
                choices.append(comp)

        if obj == 'T0':
            choices.append(obj)
            pos = num_of_choices
        else:
            choices.pop()
            pos = random.randint(0, num_of_choices-1)
            choices.insert(pos, obj)
            choices.append('T0')

        assert len(choices) == num_of_choices+1

        return choices, pos

    def _add_empty_links(components, links):

        for c in components.keys():
            if c in links.keys():
                continue
            else:
                links[c]='T0'

        placeholder = 'zzz'
        components['T0'] = placeholder

        return components, links

    for f in os.listdir(input_dir):
        if '.ann' in f:
            links = {}
            relations = {}
            components = {}
            f_name = f[:-4]

            # open ann file and get components and relations
            with open(input_dir + f_name + '.ann', 'r', encoding='utf-8') as fann:
                lines = fann.read().splitlines()
                for line in lines:
                    line = line.split('\t')
                    if 'T' in line[0]:
                        components[line[0]] = line[2].replace('\n','')
                    elif 'R' in line[0]:
                        rel = line[1].split()[1:]
                        links[rel[0][5:]] = rel[1][5:]
                        relations[rel[0][5:]] = line[1].split()[0]


            assert len(relations.keys()) == len(links.keys())

            components, links = _add_empty_links(components, links)
            full_questions = []

            if len(components.keys()) < 3:
                continue

            for subj in links.keys():
                choices, label = _build_choices((subj, links[subj]), list(components.keys()))

                #full_questions.append([subj, choices, label])
                if subj in relations.keys():
                    relationlabel = relations[subj]
                else:
                    relationlabel = "NoRelation"
                full_questions.append([subj, choices, label, relationlabel])
            # append to file
            with open(output_file + '.tsv', 'a', encoding='utf-8') as fout:
                for subj, choices, label, relationlabel in full_questions:
                    fout.write(components[subj].replace('"','') + '\t')
                    for choice in choices:
                        fout.write(components[choice].replace('"','') + '\t')
                    fout.write(str(label) + "\t" + relationlabel + '\n')

            # append to file
            with open(output_file+'_gold_labels' + '.tsv', 'a', encoding='utf-8') as fout:

                for rel in relations.keys():
                    subj = components[rel].replace('"','')
                    obj = components[links[rel]].replace('"','')
                    label = relations[rel]
                    fout.write( subj + '\t' + obj + '\t' + label + '\n')

    print('DONE WRITING')


def main():

    # input_dir = '../data/glaucoma_test/glaucoma/'
    # input_dir = '../data/neoplasm/neo_test/'
    # output = '../data/glaucoma'
    # output = '../data/neoplasm/test_relations'

    input_dir = '../data/cdcp/relations/dev/'
    output = '../data/cdcp/relations/'

    # input_dir = '../data/test/'
    # output = '../data/test/'

    # create_tsv_for_relation_classification('../data/cdcp/brat/dev/', '../data/cdcp/relations/dev_relations')
    # create_tsv_for_relation_classification('../data/cdcp/brat/train/', '../data/cdcp/relations/train_relations')
    # create_tsv_for_relation_classification('../data/cdcp/brat/test/', '../data/cdcp/relations/test_relations')

    # create_tsv_for_relation_classificationCDCP('../data/cdcp/brat/dev/', '../data/cdcp/relations/dev_relations')
    # create_tsv_for_relation_classificationCDCP('../data/cdcp/brat/train/', '../data/cdcp/relations/train_relations')
    # create_tsv_for_relation_classificationCDCP('../data/cdcp/brat/test/', '../data/cdcp/relations/test_relations')

    create_tsv_for_relation_classificationCDCP('../data/cdcp/brat/dev/',
                                               '../data/cdcp/relations_test_sample/dev_relations',
                                               sample_for_testing=True)
    create_tsv_for_relation_classificationCDCP('../data/cdcp/brat/train/',
                                               '../data/cdcp/relations_test_sample/train_relations',
                                               sample_for_testing=True)
    create_tsv_for_relation_classificationCDCP('../data/cdcp/brat/test/',
                                               '../data/cdcp/relations_test_sample/test_relations',
                                               sample_for_testing=True)
    # create_tsv_for_multiple_choice(input_dir, output)

if __name__ == "__main__":
    main()