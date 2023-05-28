## Wrapper for the parser by Zhang et al. (2021)

[The source](https://github.com/NLP-Discourse-SoochowU/sota_end2end_parser). 

### Docker

[Docker image for a remote isanlp processor](https://hub.docker.com/r/tchewik/isanlp_zhang21).

### Local run

1. Download the weights into `data/models_saved/model.pth` and `data/models_saved/xl_model.pth`. [Link 1 (Baidu)](https://github.com/NLP-Discourse-SoochowU/sota_end2end_parser/blob/main/README.md), [link 2 (Dropbox)](https://github.com/NLP-Discourse-SoochowU/sota_end2end_parser/issues/2).
2. Configure the environment: `source setup_environment.sh`. If necessary, install the environment as a jupyter kernel: `pip install ipykernel && python -m ipykernel install --name "rst_parser_zhang"`
3. Use in the code:
```python
from isanlp import PipelineCommon
from isanlp.processor_udpipe import ProcessorUDPipe
from isanlp_processor import ProcessorRST


ppl = PipelineCommon([
    #(ProcessorSpaCy(model_name='en_core_web_sm', morphology=True, parser=False, ner=False, delay_init=False),
    (ProcessorUDPipe(model_path='data/english-ewt-ud-2.5-191206.udpipe', parser=False),
     ['text'],
     {'tokens': 'tokens',
      'sentences': 'sentences'}),
    (ProcessorRST(),
     ['text', 'tokens', 'sentences'],
     {0: 'rst'})
])

some_text = "Brown fox jumped over the tree because it had to."
result = ppl(some_text)
```

```python
>>> print(result['rst'][0])
id: 1
text: Brown fox jumped over the tree because it had to.
proba: 1.0
relation: Explanation
nuclearity: NS
left: Brown fox jumped over the tree
right: because it had to.
start: 0
end: 49
```