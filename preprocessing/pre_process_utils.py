import json


# From Marseille
class CDCPArgumentationDoc:

    def __init__(self, file_root, merge_consecutive_spans=True):

        self.doc_id = int(file_root[-5:])
        self._ann_path = file_root + ".ann.json"

        with open(file_root + ".txt") as f:
            self.raw_text = f.read()

        # annotation is always loaded
        try:
            with open(self._ann_path, encoding="utf8") as f:
                ann = json.load(f)

            self.url = {int(key): val for key, val in ann['url'].items()}
            self.prop_labels = ann['prop_labels']
            self.prop_offsets = [(int(a), int(b))
                                 for a, b in ann['prop_offsets']]
            self.reasons = [((int(a), int(b)), int(c), 'reason')
                            for (a, b), c in ann['reasons']]
            self.evidences = [((int(a), int(b)), int(c), 'evidence')
                              for (a, b), c in ann['evidences']]

            self.links = self.reasons + self.evidences

        except FileNotFoundError:
            raise FileNotFoundError("Annotation json not found at {}"
                                    .format(self._ann_path))

        if merge_consecutive_spans:
            merge_spans(self)

        self.links = _transitive(self.links)
        self.links_dict = {a: {'link': b, 'type': l_type} for (a, b, l_type) in self.links}
        self.reasons = [(a, b) for (a, b, l_type) in self.links if l_type == 'reason']
        self.evidences = [(a, b) for (a, b, l_type) in self.links if l_type == 'evidence']


# From Marseille
def merge_spans(doc, include_nonarg=True):
    """Normalization needed for CDCP data because of multi-prop spans"""

    # flatten multi-prop src spans like (3, 6) into new propositions
    # as long as they never overlap with other links. This inevitably will
    # drop some data but it's a very small number.

    # function fails if called twice because
    #    precondition: doc.links = [((i, j), k)...]
    #    postcondition: doc.links = [(i, k)...]

    new_links = []
    new_props = {}
    new_prop_offsets = {}

    dropped = 0

    for (start, end), trg, l_type in doc.links:

        if start == end:
            new_props[start] = (start, end)
            new_prop_offsets[start] = doc.prop_offsets[start]

            new_props[trg] = (trg, trg)
            new_prop_offsets[trg] = doc.prop_offsets[trg]

            new_links.append((start, trg, l_type))

        elif start < end:
            # multi-prop span. Check for problems:

            problems = []
            for (other_start, other_end), other_trg, other_l_type in doc.links:
                if start == other_start and end == other_end:
                    continue

                # another link coming out of a subset of our span
                if start <= other_start <= other_end <= end:
                    problems.append(((other_start, other_end), other_trg))

                # another link coming into a subset of our span
                if start <= other_trg <= end:
                    problems.append(((other_start, other_end), other_trg))

            if not len(problems):
                if start in new_props:
                    assert (start, end) == new_props[start]

                new_props[start] = (start, end)
                new_prop_offsets[start] = (doc.prop_offsets[start][0],
                                           doc.prop_offsets[end][1])

                new_props[trg] = (trg, trg)
                new_prop_offsets[trg] = doc.prop_offsets[trg]

                new_links.append((start, trg, l_type))

            else:
                # Since we drop the possibly NEW span, there is no need
                # to remove any negative links.
                dropped += 1

    if include_nonarg:
        used_props = set(k for a, b in new_props.values()
                         for k in range(a, b + 1))
        for k in range(len(doc.prop_offsets)):
            if k not in used_props:
                new_props[k] = (k, k)
                new_prop_offsets[k] = doc.prop_offsets[k]

    mapping = {key: k for k, key in enumerate(sorted(new_props))}
    props = [val for _, val in sorted(new_props.items())]
    doc.prop_offsets = [val for _, val in sorted(new_prop_offsets.items())]
    doc.links = [(mapping[src], mapping[trg], l_type) for src, trg, l_type in new_links]

    doc.prop_labels = [merge_prop_labels(doc.prop_labels[a:1 + b])
                       for a, b in props]

    return doc


# From Marseille
def merge_prop_labels(labels):
    """After joining multiple propositions, we need to decide the new type.

    Rules:
        1. if the span is a single prop, keep the label
        2. if the span props have the same type, use that type
        3. Else, rules from Jon: policy>value>testimony>reference>fact
    """

    if len(labels) == 1:
        return labels[0]

    labels = set(labels)

    if len(labels) == 1:
        return next(iter(labels))

    if 'policy' in labels:
        return 'policy'
    elif 'value' in labels:
        return 'value'
    elif 'testimony' in labels:
        return 'testimony'
    elif 'reference' in labels:
        return 'reference'
    elif 'fact' in labels:
        return 'fact'
    else:
        raise ValueError("weird labels: {}".format(" ".join(labels)))


# From Marseille
def _transitive(links):
    """perform transitive closure of links.

    For input [(1, 2), (2, 3)] the output is [(1, 2), (2, 3), (1, 3)]
    """

    links = set(links)
    while True:
        new_links = [(src_a, trg_b, l_type_a)
                     for src_a, trg_a, l_type_a in links
                     for src_b, trg_b, l_type_b in links
                     if trg_a == src_b
                     and l_type_a == l_type_b
                     and (src_a, trg_b, l_type_a) not in links]
        if new_links:
            links.update(new_links)
        else:
            break

    return links
