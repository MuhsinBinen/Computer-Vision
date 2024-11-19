
from roboflow import Roboflow
rf = Roboflow(api_key="z4Tswq31URmsPqJbHdjO")
project = rf.workspace("computervisionprojects-eapsv").project("objectdetectionproject-lrctu")
version = project.version(2)
dataset = version.download("yolov8")