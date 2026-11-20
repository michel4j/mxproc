import re
from pathlib import Path
from parsefire import parser


DATA_PATH = Path(__file__).parent / "data"


class XDSParser(parser.TextParser):
    LEXICON = {
        "COLSPOT.LP": DATA_PATH / "spots.yml",
        "IDXREF.LP": DATA_PATH / "idxref.yml",
        "XPLAN.LP": DATA_PATH / "xplan.yml",
        "INTEGRATE.LP": DATA_PATH / "integrate.yml",
        "CORRECT.LP": DATA_PATH / "correct.yml",
        "XSCALE.LP": DATA_PATH / "xscale.yml",
        "XDSSTAT.LP": DATA_PATH / "xdsstat.yml",
        "XPARM.XDS": DATA_PATH / "xparm.yml",
        "GXPARM.XDS": DATA_PATH / "xparm.yml",
        "XTRIAGE.LP": DATA_PATH / "xtriage.yml"
    }

    @classmethod
    def parse_xscale(cls, work_dir: Path = Path(".")):
        """
        Harvest data from XSCALE
        """
        xscale_lp = Path(work_dir) / 'XSCALE.LP'
        if not xscale_lp.exists():
            return {}

        with open(xscale_lp, 'r') as handle:
            data = handle.read()

        # extract separate sections corresponding to different datasets
        header = "\n".join(re.findall(r'(CONTROL CARDS.+?CORRECTION FACTORS AS FUNCTION)', data, re.DOTALL))
        section = re.compile(
            r'(STATISTICS OF SCALED OUTPUT DATA SET :\s+(\S+).+? ========== )STATISTICS OF INPUT DATA SET', re.DOTALL
        )
        footer = re.compile(
            r'(WILSON STATISTICS OF SCALED DATA SET:\s+(\S+).+?HIGHER ORDER MOMENTS OF WILSON)', re.DOTALL
        )
        footers = {key: body for body, key in footer.findall(data)}

        data_sections = {
            key: "\n".join([header, body, footers[key]])
            for body, key in section.findall(data)
        }

        lexicon = cls.get_lexicon('XSCALE.LP')
        return {
            key: cls.parse_text(info, lexicon)
            for key, info in data_sections.items()
        }
