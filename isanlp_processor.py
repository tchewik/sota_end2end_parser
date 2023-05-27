from pipeline import *
from config import *
from isanlp.annotation_rst import DiscourseUnit
from isanlp.annotation import Span


class ProcessorRST:
    def __init__(self,
                 path_segmenter='data/seg/models_saved/EN_200.model',
                 path_parser='data/models_saved/model.pth',
                 cuda_id=-1):

        self.segmenter = torch.load(path_segmenter, map_location=torch.device('cpu'))
        self.segmenter.eval()

        self.model = torch.load(path_parser, map_location=torch.device('cpu'))
        self.model.eval()

        if cuda_id != -1:
            self.segmenter.cuda(cuda_id)
            self.model.cuda(cuda_id)

        self.parser = PartitionPtrParser()

    def __call__(self, annot_text, annot_tokens, annot_sentences, *args, **kwargs):
        # segmenting
        sents_dt = self._prep_seg(text=annot_text, tokens=annot_tokens, sentences=annot_sentences)
        seg_edus = self._do_seg(sents_dt)

        # parsing
        trees = self.do_parse(seg_edus)

        # converting to isanlp.DiscourseUnit objects
        self._node_id = 0
        trees = [self._convert_to_isanlp(tree=tree, text=annot_text, tokens=annot_tokens) for tree in trees]

        return trees

    def _prep_seg(self, text, tokens, sentences):
        """ Overrides the original function from pipeline.py """

        sentences = [text[tokens[sent.begin].begin:tokens[sent.end-1].end] for sent in sentences]

        sents_dt = []
        for idx, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) == 0:
                continue
            tok_pairs = nlp.pos_tag(sent.strip())
            words = [pair[0] for pair in tok_pairs]
            tags = [pair[1] for pair in tok_pairs]
            word_ids = []
            for word in words:
                if word.lower() in word2ids.keys():
                    word_ids.append(word2ids[word.lower()])
                else:
                    word_ids.append(UNK_ids)
            pos_ids = [pos2ids[tag] for tag in tags]

            word_ids.insert(0, PAD_ID)
            pos_ids.insert(0, PAD_ID)

            graph_ids = []
            dependency = nlp.dependency_parse(sent)
            # (type, "head", "dep")
            for i, dep_pair in enumerate(dependency):
                graph_ids.append((i, i, sync2ids["self"]))
                graph_ids.append((dep_pair[1], dep_pair[2], sync2ids["head"]))
                graph_ids.append((dep_pair[2], dep_pair[1], sync2ids["dep"]))
            elmo_ids = batch_to_ids([words])
            tmp_sent_tokens_emb = elmo(elmo_ids)["elmo_representations"][0][0]
            tmp_sent_tokens_emb = torch.cat((ELMO_ROOT_PAD, tmp_sent_tokens_emb), 0)
            sents_dt.append((words, word_ids, pos_ids, graph_ids, None, tmp_sent_tokens_emb))

        return sents_dt

    def _do_seg(self, sents_dt_):
        result_dt = [sents_dt_]

        # segment
        edus_all = []
        for doc_dt in result_dt:
            batch_iter = gen_batch_iter(doc_dt, batch_s=1)
            for n_batch, inputs in enumerate(batch_iter, start=1):
                words_all, word_ids, word_elmo_embeddings, pos_ids, graph, masks = inputs
                pred = self.segmenter.predict_(word_ids, word_elmo_embeddings, pos_ids, graph, masks)
                predict = pred.data.cpu().numpy()
                # transform to EDUs
                words_all = words_all[0]
                edus_all += fetch_edus(words_all, predict)
            edus_all.append("")

        return edus_all

    @staticmethod
    def collect_all_leafs(node):
        if not node:
            return []

        if node.temp_edu and type(node.temp_edu) == str:
            return [node.temp_edu]

        elif type(node.temp_edu) == tuple:
            return [node.temp_edu[0]]

        return ProcessorRST.collect_all_leafs(node.left_child) + ProcessorRST.collect_all_leafs(node.right_child)

    @staticmethod
    def find_tokens_in_text(tokenized_text, original_tokens, start=0):
        """ Tokenized text: text from the rst parser
            Original tokens: list[isanlp.Token] """

        str_tokens = tokenized_text.split()
        chr_tokens = ''.join(str_tokens)

        for begin in range(start, len(original_tokens) - 1):
            for end in range(begin, len(original_tokens) + 1):
                new_text = ''.join([token.text for token in original_tokens[begin:end] if token.text.strip()])
                new_text = ''.join(new_text.split())
                if new_text == chr_tokens:
                    return original_tokens[begin].begin, original_tokens[end - 1].end

        return 0, original_tokens[-1].end

    def do_parse(self, seg_edus):
        edus = prepare_dt(seg_edus)

        trees = []
        for idx, doc_instances in enumerate(edus):
            tree = self.parser.parse(doc_instances, self.model)
            trees.append(tree)
        return trees

    def _convert_to_isanlp(self, tree, text, tokens, start=0):
        self._node_id += 1
        _current_id = self._node_id

        _text = ' '.join(ProcessorRST.collect_all_leafs(tree))
        start, end = ProcessorRST.find_tokens_in_text(_text, tokens, start=start)

        if not tree.left_child:
            return DiscourseUnit(id=_current_id,
                                 start=start,
                                 end=end,
                                 text=text[start:end],
                                 relation='elementary')

        left_unit = self._convert_to_isanlp(tree=tree.left_child, text=text, tokens=tokens)
        right_unit = self._convert_to_isanlp(tree=tree.right_child, text=text, tokens=tokens)

        return DiscourseUnit(id=_current_id,
                             start=start,
                             end=end,
                             left=left_unit, right=right_unit,
                             text=text[start:end],
                             relation=tree.child_rel,
                             nuclearity=tree.child_NS_rel)
