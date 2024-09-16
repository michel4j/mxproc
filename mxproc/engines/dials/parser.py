from pathlib import Path
from parsefire import parser


DATA_PATH = Path(__file__).parent / "data"


class DIALSParser(parser.TextParser):
    LEXICON = {
        "dials.import.log": DATA_PATH / "import.yml",
        "dials.find_spots.log": DATA_PATH / "spots.yml",
        "dials.index.log": DATA_PATH / "index.yml",
    }
