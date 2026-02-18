


from pathlib import Path

class DatasetCardCreator:
    def __init__(self, sections: dict, renderer):
        self.sections = sections
        self.renderer = renderer

    def generate(self, output_path: str = "DATASET_CARD.md"):
        text = self.renderer.render(self.sections)
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"Dataset card created at: {output_path}")
        
        
        