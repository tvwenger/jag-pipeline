from setuptools import setup
import re


def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


setup(
    name="jag-pipeline",
    version=get_property("__version__", "jagpipe"),
    description="DRAO JAG data reduction pipeline",
    author="Trey V. Wenger",
    packages=["jagpipe"],
    install_requires=["numpy", "astropy", "h5py"],
    entry_points={
        "console_scripts": [
            "jagpipe-combine=jagpipe.combine:main",
            "jagpipe-flagchan=jagpipe.flagchan:main",
        ]
    },
)
