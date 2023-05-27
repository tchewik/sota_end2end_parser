from isanlp_processor import ProcessorRST
from isanlp import PipelineCommon


def create_pipeline(delay_init):
    pipeline_default = PipelineCommon([(ProcessorRST(),
                                        ['text', 'tokens', 'sentences'],
                                        {0: 'rst'})
                                       ],
                                      name='default')

    return pipeline_default
