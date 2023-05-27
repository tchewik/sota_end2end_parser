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
from isanlp.processor_spacy import ProcessorSpaCy
from isanlp_processor import ProcessorRST

ppl = PipelineCommon([
    (ProcessorSpaCy(model_name='en_core_web_sm', morphology=True, parser=False, ner=False, delay_init=False),
     ['text'],
     {'tokens': 'tokens',
      'sentences': 'sentences'}),
    (ProcessorRST(),
     ['text', 'tokens', 'sentences'],
     {'rst': 'rst'})
])

result = ppl(some_text)
```