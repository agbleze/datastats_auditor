

#%%
from pathlib import Path

class DatasetCardCreator:
    def __init__(self, sections: dict, renderer):
        self.sections = sections
        self.renderer = renderer

    def generate(self, output_path: str = "DATASET_CARD.md"):
        text = self.renderer.render(self.sections)
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"Dataset card created at: {output_path}")
        
        




def build_motivations_and_use(
    purposes=None,
    domain_applications=None,
    problem_space=None,
    primary_motivations=None,
    intended_tasks=None,
    intended_objects=None,
):
    """
    Dynamically generates the 'Motivations & Use' section of the dataset card.
    """

    purposes = purposes or ["Training", "Validation", "Research", "Evaluation"]
    domain_applications = domain_applications or ["Machine Learning", "Computer Vision", "Object Recognition"]
    problem_space = problem_space or (
        "This dataset was created to support research, development, and experimentation "
        "in real-world industrial settings with constraints such as camera angle, orientation, "
        "lighting conditions, and environmental variations that are difficult to replicate "
        "in synthetic simulations."
    )
    primary_motivations = primary_motivations or (
        "Capture realistic and representative variations in annotations, foreground/background "
        "composition, and scene interactions within the target environments."
    )
    intended_tasks = intended_tasks or ["Object detection", "Person detection"]
    intended_objects = intended_objects or []

    # Format lists into bullet points
    def bullets(items):
        return "\n".join([f"- {item}" for item in items])

    section = f"""
# Motivations & Use

## Dataset Purpose(s)
{bullets(purposes)}

## Key Domain Application(s)
{bullets(domain_applications)}

## Problem Space
{problem_space}

## Primary Motivation(s)
{primary_motivations}

## Intended Use Case(s)

### Tasks
{bullets(intended_tasks)}

### Objects Labelled
{bullets(intended_objects) if intended_objects else "- (No specific objects selected)"}
"""
    return section.strip()        



#%%

class MarkdownRenderer:
    def render(self, sections: dict) -> str:
        md = "# 📘 Dataset Card\n\n"
        for title, content in sections.items():
            if content is None or not str(content).strip():
                continue
            md += f"## {title}\n\n{content.strip()}\n\n"
        return md
    
#%%

content = build_motivations_and_use(
    intended_tasks=["Object detection", "Fairness evaluation"],
    intended_objects=["Person", "Helmet", "Safety vest"],
)
print(content)
# %%
motsect = {"Motivation and Intended Use": content}

mkd_render = MarkdownRenderer()

cardgen = DatasetCardCreator(sections=motsect, renderer=mkd_render)
# %%
cardgen.generate()
# %%
