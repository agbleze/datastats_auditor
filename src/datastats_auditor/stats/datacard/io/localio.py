from ..core.io.baseio import BaseCardExporter
from ..utils import write_md_to_pdf


class LocalCardExporter(BaseCardExporter):
    name = "local_card_exporter"
    status = "stable"
    
    def __init__(self, markdown_content, pdf_path):
        self.markdown_content = markdown_content
        self.pdf_path = pdf_path
        
    def export(self):
        write_md_to_pdf(markdown_content=self.markdown_content,
                        pdf_path=self.pdf_path
                        )
