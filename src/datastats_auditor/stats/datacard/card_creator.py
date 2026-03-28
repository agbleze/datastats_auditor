from .base_card_creator import BaseCardCreator
from .datacard_generator import create_data_card


class CardCreator(BaseCardCreator):
    name = "card_creator"
    status = "stable"
    
    def __init__(self, split_stats_result,
                 drift_result, version_id,
                 card_name,
                 intended_objects=None,
                 **kwargs
                 ):
        self.split_stats_result = split_stats_result
        self.drift_result = drift_result
        self.version_id = version_id
        self.card_name = card_name
        self.intended_objects = intended_objects
        
    def create_card(self, **kwargs):
        card_md_content = create_data_card(split_stats_result=self.split_stats_result,
                                            drift_result=self.drift_result,
                                            version_id=self.version_id, 
                                            card_name=self.card_name,
                                            intended_objects=self.intended_objects,
                                            **kwargs
                                            )
        return card_md_content